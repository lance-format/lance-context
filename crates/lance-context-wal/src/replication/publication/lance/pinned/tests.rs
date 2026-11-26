use super::*;
use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use futures::TryStreamExt;
use lance::{
    dataset::{builder::DatasetBuilder, CommitBuilder, WriteParams},
    index::DatasetIndexExt,
    Dataset,
};
use lance_index::{
    scalar::{BuiltinIndexType, ScalarIndexParams},
    IndexType,
};
use lance_table::{
    io::commit::ConditionalPutCommitHandler,
    transaction::{Operation, Transaction as NativeTransaction},
};
use object_store::ObjectStoreExt;
use std::collections::BTreeMap;

fn binding() -> (Layout, Uuid, Uuid) {
    (
        Layout {
            cluster: Uuid::new_v4(),
            nodes: (1..=3).map(|id| (id, format!("wal-{id}:7800"))).collect(),
            groups: BTreeMap::from([(10, vec![1, 2, 3])]),
            buckets: vec![10],
        },
        Uuid::new_v4(),
        Uuid::new_v4(),
    )
}

#[tokio::test]
async fn pinned_staging_manifest_opens_and_supports_explicit_detached_derivation() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = root.path().to_str().unwrap();
    let batch =
        RecordBatch::try_from_iter([("value", Arc::new(Int32Array::from(vec![3, 5, 8])) as _)])?;
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        uri,
        Some(WriteParams {
            enable_v2_manifest_paths: true,
            ..Default::default()
        }),
    )
    .await?;
    let version = dataset.version().version;
    let session = Arc::new(lance::session::Session::default());
    let (store, base) =
        ObjectStore::from_uri_and_params(session.store_registry(), uri, &Default::default())
            .await?;
    let canonical = ManifestNamingScheme::V2.manifest_path(&base, version);
    let staged = Path::parse(format!("{canonical}-{}", Uuid::new_v4()))?;
    store.inner.rename(&canonical, &staged).await?;
    let (layout, table, incarnation) = binding();
    let record = ManifestRecord {
        version,
        identity: manifest::identity_for_binding(
            layout.cluster,
            layout.fingerprint()?,
            table,
            incarnation,
            version,
            17,
        ),
        object: ManifestObject {
            path: format!("_versions/{}", staged.filename().unwrap()),
            bytes: store.inner.head(&staged).await?.size,
            v2_names: true,
        },
        transaction: Uuid::parse_str(&dataset.read_transaction().await?.unwrap().uuid)?,
        index: 17,
        plan_index: 0,
        prepared_index: 0,
        descriptor_digest: [0; 32],
    };
    // Raw opens cannot resolve the unique staged object even with its version.
    assert!(DatasetBuilder::from_uri(uri)
        .with_version(version)
        .load()
        .await
        .is_err());
    let handler = Arc::new(PinnedManifest::new(
        base.clone(),
        layout.cluster,
        layout.fingerprint()?,
        table,
        incarnation,
        &record,
    )?);
    let mut selected = DatasetBuilder::from_uri(uri)
        .with_commit_handler(handler.clone())
        .with_session(session)
        .with_version(version)
        .load()
        .await?;
    assert_eq!(selected.count_rows(None).await?, 3);
    let observed = selected
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(observed, vec![batch]);
    assert!(DatasetBuilder::from_uri(uri)
        .with_commit_handler(handler.clone())
        .with_version(version + 1)
        .load()
        .await
        .is_err());
    assert!(handler
        .version_exists(
            &base,
            version + 1,
            store.inner.as_ref(),
            ManifestNamingScheme::V2
        )
        .await
        .is_err());
    assert!(handler
        .resolve_latest_location(&Path::from("foreign"), &store)
        .await
        .is_err());
    assert!(handler
        .list_manifest_locations(&base, &store, false)
        .try_collect::<Vec<_>>()
        .await
        .is_err());
    assert!(handler.delete(&base).await.is_err());
    assert!(handler
        .forget_version(&base, version, &record.identity.to_string())
        .await
        .is_err());

    // The pinned reader itself cannot publish or grant write authority. A
    // separately authorized derived writer explicitly selects its own handler.
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
    let index = selected
        .create_index_builder(&["value"], IndexType::ZoneMap, &params)
        .name("values".into())
        .execute_uncommitted()
        .await?;
    let index_id = index.uuid;
    let transaction = NativeTransaction::new(
        version,
        Operation::CreateIndex {
            new_indices: vec![index],
            removed_indices: vec![],
        },
        None,
    );
    assert!(CommitBuilder::new(Arc::new(selected.clone()))
        .with_detached(true)
        .with_max_retries(1)
        .execute(transaction.clone())
        .await
        .is_err());
    let detached = CommitBuilder::new(Arc::new(selected.clone()))
        .with_commit_handler(Arc::new(ConditionalPutCommitHandler))
        .with_detached(true)
        .with_max_retries(1)
        .execute(transaction)
        .await?;
    assert!(detached.version().version >= 1u64 << 63);
    let first_version = detached.version().version;
    // CommitBuilder's write override does not replace the returned Dataset's
    // inherited read handler. Reopen before continuing a derived index chain.
    assert!(detached.checkout_version(first_version).await.is_err());
    let mut reopened = DatasetBuilder::from_uri(uri)
        .with_version(first_version)
        .load()
        .await?;
    assert_eq!(reopened.count_rows(None).await?, 3);
    assert_eq!(reopened.load_indices().await?[0].uuid, index_id);
    let rows = reopened
        .scan()
        .filter("value >= 5")?
        .try_into_batch()
        .await?;
    assert_eq!(rows.num_rows(), 2);

    let bloom_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter);
    let bloom = reopened
        .create_index_builder(&["value"], IndexType::BloomFilter, &bloom_params)
        .name("values_bloom".into())
        .execute_uncommitted()
        .await?;
    let second = CommitBuilder::new(Arc::new(reopened))
        .with_commit_handler(Arc::new(ConditionalPutCommitHandler))
        .with_detached(true)
        .execute(NativeTransaction::new(
            first_version,
            Operation::CreateIndex {
                new_indices: vec![bloom],
                removed_indices: vec![],
            },
            None,
        ))
        .await?;
    let chain = DatasetBuilder::from_uri(uri)
        .with_version(second.version().version)
        .load()
        .await?;
    assert_eq!(chain.load_indices().await?.len(), 2);
    assert_eq!(
        chain
            .checkout_version(first_version)
            .await?
            .load_indices()
            .await?[0]
            .uuid,
        index_id
    );
    let last = chain
        .read_version_transaction(chain.version().version)
        .await?
        .transaction
        .unwrap();
    assert!(matches!(last.operation, Operation::CreateIndex { .. }));
    assert_eq!(last.read_version, first_version);
    let previous = chain
        .read_version_transaction(first_version)
        .await?
        .transaction
        .unwrap();
    assert!(matches!(previous.operation, Operation::CreateIndex { .. }));
    assert_eq!(previous.read_version, version);
    // Stop ordinary ancestor traversal at the selected base. Its unique
    // staging path is resolved only through the independently pinned handle.
    assert!(chain.checkout_version(version).await.is_err());
    assert!(chain.read_version_transaction(version).await.is_err());
    let selected_transaction = selected
        .read_version_transaction(version)
        .await?
        .transaction
        .unwrap();
    assert_eq!(selected_transaction.uuid, record.transaction.to_string());
    assert_eq!(
        selected
            .checkout_version(version)
            .await?
            .count_rows(None)
            .await?,
        3
    );
    // Publishing the detached index cannot move the selected body snapshot.
    assert!(DatasetBuilder::from_uri(uri)
        .with_commit_handler(handler.clone())
        .with_version(version)
        .load()
        .await?
        .load_indices()
        .await?
        .is_empty());
    assert_eq!(
        handler
            .resolve_latest_location(&base, &store)
            .await?
            .version,
        version
    );

    for kind in 0..4 {
        let mut invalid = record.clone();
        match kind {
            0 => invalid.identity = Uuid::new_v4(),
            1 => invalid.object.path = "../foreign.manifest".into(),
            2 => invalid.version += 1,
            _ => invalid.index += 1,
        }
        assert!(PinnedManifest::new(
            base.clone(),
            layout.cluster,
            layout.fingerprint()?,
            table,
            incarnation,
            &invalid
        )
        .is_err());
    }
    assert!(PinnedManifest::new(
        base.clone(),
        layout.cluster,
        layout.fingerprint()?,
        Uuid::new_v4(),
        incarnation,
        &record
    )
    .is_err());
    assert!(PinnedManifest::new(
        base,
        layout.cluster,
        layout.fingerprint()?,
        table,
        Uuid::new_v4(),
        &record
    )
    .is_err());
    Ok(())
}
