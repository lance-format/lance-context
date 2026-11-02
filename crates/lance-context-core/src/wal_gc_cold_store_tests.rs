use super::*;
use lance_context_api::{AddRecordRequest, DatagenEventDto, DatagenStoreApi};
use object_store::ObjectStoreExt;

#[tokio::test]
async fn retain_uri_discovery_does_not_open_storage() {
    discover_uri(Mode::Retain, "unsupported://must-not-be-opened")
        .await
        .unwrap();
}

#[tokio::test]
async fn cold_context_and_datagen_retirements_recover_without_dataset_open() {
    // The test runner starts in retain mode, so writers persist intents but
    // cannot collect. Closing and dropping them loses all remembered tickets.
    assert_eq!(Mode::configured().unwrap(), Mode::Retain);
    for (kind, through_root) in [
        ("context", false),
        ("datagen", false),
        ("context", true),
        ("datagen", true),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().join(format!("cold.{kind}.lance"));
        let uri = uri.to_str().unwrap();
        if kind == "context" {
            let mut writer = crate::ContextStore::open(uri).await.unwrap();
            let record = crate::record_from_add_request(
                &AddRecordRequest {
                    text_payload: Some("retained context payload".into()),
                    ..Default::default()
                },
                "record".into(),
                "run".into(),
            );
            writer.add(&[record]).await.unwrap();
            assert_eq!(writer.cleanup_wal().await.unwrap(), 1);
            assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
            writer.close().await.unwrap();
        } else {
            let mut writer = crate::DatagenStore::open(uri).await.unwrap();
            let event: DatagenEventDto = serde_json::from_value(serde_json::json!({
                "event_id":"created", "item_id":"root", "root_item_id":"root",
                "item_seq":0, "checkpoint_id":"created", "event_type":"ITEM_CREATED",
                "run_id":"run", "writer_epoch":"writer", "status":"running",
                "schema_version":crate::DATAGEN_SCHEMA_VERSION
            }))
            .unwrap();
            DatagenStoreApi::append(&mut writer, &[event])
                .await
                .unwrap();
            assert_eq!(writer.cleanup_own_shard().await.unwrap(), 1);
            assert_eq!(writer.pending_wal_generations().await.unwrap(), 0);
            writer.close().await.unwrap();
        }
        let (store, root) = ObjectStore::from_uri(uri).await.unwrap();
        let intents: Vec<_> = store
            .inner
            .list(Some(&root.clone().join("_wal_retirements_v1")))
            .try_collect()
            .await
            .unwrap();
        assert_eq!(intents.len(), 1);
        let generation_files: Vec<_> = store
            .inner
            .list(Some(&root.clone().join("_mem_wal")))
            .try_filter(|meta| {
                let path = meta.location.to_string();
                futures::future::ready(path.contains("_gen_") || path.contains("/gen_"))
            })
            .try_collect()
            .await
            .unwrap();
        assert!(!generation_files.is_empty());

        discover_uri(Mode::Retain, uri).await.unwrap();
        assert!(store.inner.head(&intents[0].location).await.is_ok());
        // No typed store, remembered ticket, or pending-generation trigger is
        // supplied to the new recovery admission.
        if through_root {
            root_discovery::discover_root(Mode::Collect, directory.path().to_str().unwrap())
                .await
                .unwrap();
        } else {
            discover_uri(Mode::Collect, uri).await.unwrap();
        }
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                match store.inner.head(&intents[0].location).await {
                    Err(object_store::Error::NotFound { .. }) => break,
                    Ok(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                    Err(error) => panic!("retirement observation failed: {error}"),
                }
            }
        })
        .await
        .unwrap();
        for file in generation_files {
            assert!(matches!(
                store.inner.head(&file.location).await,
                Err(object_store::Error::NotFound { .. })
            ));
        }
        if kind == "context" {
            let reader = crate::ContextStore::open(uri).await.unwrap();
            assert_eq!(
                reader.get("record").await.unwrap().unwrap().text_payload,
                Some("retained context payload".into())
            );
        } else {
            let reader = crate::DatagenStore::open(uri).await.unwrap();
            let events = reader.events_for_item("root").await.unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].event_id, "created");
        }
    }
}
