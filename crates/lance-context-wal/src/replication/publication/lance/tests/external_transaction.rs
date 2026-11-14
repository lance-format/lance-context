use super::*;
use prost::Message;

#[tokio::test]
async fn external_transaction_identity_is_checked_before_upload_or_reservation() -> Result<()> {
    // Exact filename with foreign protobuf identity, foreign path, malformed
    // metadata and over-budget metadata must all retain the original sources.
    for case in 0..6 {
        let (handler, catalog, store) = fixture()?;
        let scope = handler.scope.as_ref().unwrap();
        let filename = format!("{}-{}.txn", scope.target - 1, scope.transaction);
        let path = handler.base.child("_transactions").child(filename.as_str());
        let mut transaction = lance_table::format::pb::Transaction {
            read_version: scope.target - 1,
            uuid: scope.transaction.to_string(),
            ..Default::default()
        };
        let mut manifest = Manifest::new(
            Default::default(),
            Arc::new(vec![]),
            Default::default(),
            Default::default(),
        );
        manifest.version = scope.target;
        manifest.transaction_file = Some(filename);
        match case {
            0 => transaction.uuid = Uuid::new_v4().to_string(),
            1 => transaction.read_version -= 1,
            2 => manifest.transaction_file = Some("../foreign.txn".into()),
            _ => {}
        }
        let bytes = match case {
            3 => Bytes::from_static(&[0x12, 0x80]), // Truncated UUID length.
            4 => Bytes::from(vec![0; transaction::MAX_EXTERNAL_BYTES as usize + 1]),
            _ => Bytes::from(transaction.encode_to_vec()),
        };
        store.inner.put(&path, bytes.into()).await?;
        let result = handler
            .commit(
                &mut manifest,
                None,
                &handler.base,
                &store,
                lance_table::io::commit::write_manifest_file_to_path,
                ManifestNamingScheme::V1,
                None,
            )
            .await;
        // The fixture deliberately returns Unknown for every reservation. Only
        // the valid case may upload a manifest and reach that catalog boundary.
        assert!(result.is_err());
        assert_eq!(
            catalog.reservations.lock().unwrap().len(),
            usize::from(case == 5)
        );
        let objects = store.inner.list(None).try_collect::<Vec<_>>().await?;
        assert!(objects.iter().any(|object| object.location == path));
        assert_eq!(objects.len(), if case == 5 { 2 } else { 1 });
    }
    Ok(())
}
