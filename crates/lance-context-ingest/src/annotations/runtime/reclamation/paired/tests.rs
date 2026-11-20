use super::*;
use arrow_array::RecordBatchIterator;
use lance::dataset::{transaction::Operation, transaction::Transaction, CommitBuilder};
use lance_context_core::annotation_staging::{Scope, BODY_SNAPSHOT_PROPERTY};
use std::collections::HashMap;

async fn narrow(uri: &str, reference: &BodySnapshot) -> Result<Dataset> {
    let batch = crate::tests::batch(&["A"], &["session"])?;
    let base = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            uri,
            None,
        )
        .await?,
    );
    let mut transaction = Transaction::new_from_version(
        base.version_id(),
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: None,
            field_metadata_updates: HashMap::new(),
        },
    );
    let scope = Scope {
        table_instance: reference.table_instance.clone(),
        source_group: 10,
        shard: 0,
    };
    transaction.transaction_properties = Some(Arc::new(HashMap::from([
        (
            BODY_SNAPSHOT_PROPERTY.into(),
            serde_json::to_string(reference)?,
        ),
        (
            "lance-context.annotation-cut".into(),
            serde_json::to_string(&(scope, 1u64))?,
        ),
    ])));
    Ok(CommitBuilder::new(base).execute(transaction).await?)
}

#[tokio::test]
async fn repeated_body_protection_skips_io_only_within_the_same_closed_page() -> Result<()> {
    exercise(true).await
}

#[tokio::test]
async fn receipt_only_protection_does_not_skip_required_body_file_protection() -> Result<()> {
    exercise(false).await
}

async fn exercise(first_retains_data: bool) -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().join("body.lance").to_str().unwrap().to_owned();
    let storage = Storage::resolve(
        uri.clone(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let batch = crate::tests::batch(&["A"], &["session"])?;
    let body = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        &uri,
        None,
    )
    .await?;
    let incarnation = Uuid::new_v4();
    let reference = BodySnapshot::capture(incarnation.to_string(), &body).await?;
    let narrow = narrow(
        root.path().join("narrow.lance").to_str().unwrap(),
        &reference,
    )
    .await?;
    let store = store(&storage).await?;
    let budget = MergeMemoryBudget::new(8 << 20);
    let data = storage
        .base
        .clone()
        .join("data")
        .join(body.manifest().fragments[0].files[0].path.as_str());
    let orphan = storage.base.clone().join("data").join("orphan.lance");
    let mut candidates = BTreeMap::from([(data.clone(), 1), (orphan.clone(), 1)]);
    let mut protected = Protection::default();
    protect_body(
        &mut candidates,
        &narrow,
        &storage,
        &store,
        incarnation,
        &budget,
        0,
        first_retains_data,
        &mut protected,
    )
    .await?;
    assert_eq!(candidates.contains_key(&data), !first_retains_data);
    let manifest = std::path::PathBuf::from(&uri).join(
        body.manifest_location()
            .path
            .as_ref()
            .strip_prefix(storage.base.as_ref())
            .unwrap()
            .trim_start_matches('/'),
    );
    let hidden = manifest.with_extension("held-for-test");
    std::fs::rename(&manifest, &hidden)?;
    // Removing the original manifest makes a redundant HEAD/open fail even
    // when Lance's session happens to retain other immutable metadata.
    let repeated = protect_body(
        &mut candidates,
        &narrow,
        &storage,
        &store,
        incarnation,
        &budget,
        0,
        true,
        &mut protected,
    )
    .await;
    assert_eq!(repeated.is_ok(), first_retains_data);
    // A new page must revalidate its own roots, even for the same reference.
    assert!(protect_body(
        &mut BTreeMap::from([(orphan.clone(), 1)]),
        &narrow,
        &storage,
        &store,
        incarnation,
        &budget,
        0,
        true,
        &mut Protection::default(),
    )
    .await
    .is_err());
    std::fs::rename(hidden, manifest)?;
    protect_body(
        &mut candidates,
        &narrow,
        &storage,
        &store,
        incarnation,
        &budget,
        0,
        true,
        &mut protected,
    )
    .await?;
    assert_eq!(candidates, BTreeMap::from([(orphan, 1)]));
    // Equal version numbers alone cannot authorize skipping a foreign identity.
    let foreign = BodySnapshot {
        transaction: Uuid::new_v4().to_string(),
        ..reference
    };
    let foreign_narrow = self::narrow(
        root.path().join("foreign.lance").to_str().unwrap(),
        &foreign,
    )
    .await?;
    assert!(protect_body(
        &mut candidates,
        &foreign_narrow,
        &storage,
        &store,
        incarnation,
        &budget,
        0,
        true,
        &mut protected,
    )
    .await
    .is_err());
    Ok(())
}
