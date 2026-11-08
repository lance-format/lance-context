use super::*;

#[tokio::test]
async fn an_old_cached_miss_does_not_hide_a_published_registration() -> Result<()> {
    let root = tempfile::tempdir()?;
    let batch = imported(0, 1)?;
    let schema = batch.schema();
    let base = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            root.path().join("turns.lance").to_str().unwrap(),
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await?,
    );
    let session = base.session().clone();
    let staging = AnnotationStaging::new(base, scope(), 0, Limits::default())?;
    let old = staging.capture("new").await?;
    assert!(old.resolve().await?.is_none());
    let value = TurnAnnotation {
        status: Some("reviewed".into()),
        ..Default::default()
    };
    assert_eq!(
        staging
            .register_committed(
                0,
                1,
                &[Registration {
                    turn: "new",
                    rollout_id: "session",
                    annotation: &value,
                }],
            )
            .await?,
        ApplyOutcome::Applied
    );
    let frozen = staging.freeze(uuid::Uuid::new_v4()).await?.unwrap();
    let prepared = frozen
        .prepare(&crate::MergeMemoryBudget::new(8 << 20), 8192)
        .await?;
    staging
        .install_published(&prepared, commit(&prepared).await?)
        .await?;
    session.file_metadata_cache().clear().await;
    // Populate the old snapshot's miss after installation and force the new
    // snapshot down its cold path. The current-value cache has been evicted.
    assert!(old.resolve().await?.is_none());
    assert_eq!(current(&staging, "new").await?, value);
    drop((old, prepared, frozen));
    assert_eq!(staging.observation().await.retained_bytes, 0);
    assert_eq!(staging.protected_bases().await?.datasets().count(), 1);
    Ok(())
}
