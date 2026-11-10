use super::*;
use lance::index::DatasetIndexInternalExt;

fn batch(ids: Vec<String>, status: &str) -> Result<RecordBatch> {
    let rows = ids.len();
    let mut labels = ListBuilder::new(StringBuilder::new());
    for _ in 0..rows {
        labels.values().append_value("original");
        labels.append(true);
    }
    let labels = labels.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("labels", labels.data_type().clone(), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(labels),
            Arc::new(StringArray::from(vec![status; rows])),
            Arc::new(Float32Array::from(vec![None; rows])),
        ],
    )?)
}

#[tokio::test]
async fn point_reads_check_the_whole_unindexed_tail_and_keep_the_captured_base() -> Result<()> {
    for stable in [false, true] {
        let root = tempfile::tempdir()?;
        let path = root.path().join("turns.lance");
        let seed = batch(vec!["seed".into()], "base")?;
        let mut base = Dataset::write(
            RecordBatchIterator::new(vec![Ok(seed.clone())], seed.schema()),
            path.to_str().unwrap(),
            Some(WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await?;
        base.create_index_builder(&["id"], IndexType::BTree, &ScalarIndexParams::default())
            .name("turn_id".into())
            .await?;
        let old = AnnotationStaging::new(Arc::new(base), scope(), 0, Limits::default())?;
        let held = old.capture("seed").await?;

        // Put the duplicate beyond two scan vectors. A result limit must not
        // allow an indexed match to skip the remaining, uncovered data.
        let tail = batch(
            (0..4096)
                .map(|n| format!("new-{n}"))
                .chain(std::iter::once("seed".into()))
                .collect(),
            "tail",
        )?;
        let appended = Dataset::write(
            RecordBatchIterator::new(vec![Ok(tail.clone())], tail.schema()),
            path.to_str().unwrap(),
            Some(WriteParams {
                mode: lance::dataset::WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?;
        assert_eq!(appended.unindexed_fragments("turn_id").await?.len(), 1);
        let current = AnnotationStaging::new(Arc::new(appended), scope(), 0, Limits::default())?;
        for id in ["new-0", "new-2048", "new-4095"] {
            assert_eq!(
                current.capture(id).await?.resolve().await?.unwrap(),
                TurnAnnotation {
                    labels: BTreeSet::from(["original".into()]),
                    status: Some("tail".into()),
                    score: None,
                }
            );
        }
        assert!(current.capture("missing").await?.resolve().await?.is_none());
        assert!(current
            .capture("seed")
            .await?
            .resolve()
            .await
            .unwrap_err()
            .to_string()
            .contains("duplicate annotation turn key"));
        assert_eq!(
            held.resolve().await?.unwrap().status.as_deref(),
            Some("base")
        );
    }
    Ok(())
}
