use super::*;
use crate::merge_write_scope::{CommitAuthorizer, MergeWriteScope};
use crate::StorageBaseOptions;
use arrow_array::{Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::Dataset;
use std::{future::Future, pin::Pin};
use tempfile::TempDir;

fn batch(id: &str, value: i64) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec![id])),
            Arc::new(Int64Array::from(vec![value])),
        ],
    )
    .unwrap()
}

async fn open(uri: &str, shard: &str) -> StorageBase {
    let mut options = StorageBaseOptions::new(batch("", 0).schema(), "event_id");
    options.shard_id = Some(shard.into());
    options.seal_on_put = false;
    StorageBase::open(uri, options, true).await.unwrap()
}

async fn merge(store: &mut StorageBase, append: bool) -> Result<usize> {
    let Some((manifests, manifest, mut prepared)) = store.prepare_cleanup_merge().await? else {
        return Ok(0);
    };
    if append {
        prepared.use_append_only();
    }
    store
        .commit_prepared_merge(&manifests, &manifest, prepared)
        .await
}

#[tokio::test]
async fn append_is_opt_in_and_ingest_retries_can_leave_base_duplicates() {
    for append in [false, true] {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut store = open(uri, "a").await;
        for value in [7, 7] {
            store.put(vec![batch("retry", value)]).await.unwrap();
            assert_eq!(merge(&mut store, append).await.unwrap(), 1);
        }
        let rows = Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap();
        assert_eq!(rows, if append { 2 } else { 1 });
        assert_eq!(merge(&mut store, append).await.unwrap(), 0);
        if !append {
            store.put(vec![batch("retry", 9)]).await.unwrap();
            merge(&mut store, false).await.unwrap();
            use futures::TryStreamExt;
            let rows = Dataset::open(uri)
                .await
                .unwrap()
                .scan()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
            assert_eq!(
                rows[0]
                    .column_by_name("value")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(0),
                9
            );
        }
        store.close().await.unwrap();
    }
}

#[tokio::test]
async fn append_rejects_stale_preparation_and_preserves_a_later_flush() {
    let dir = TempDir::new().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut store = open(uri, "a").await;
    store.put(vec![batch("first", 1)]).await.unwrap();
    store.flush().await.unwrap();
    let (manifests, manifest, mut first) = store.prepare_merge_if_ready(1).await.unwrap().unwrap();
    let (_, _, mut stale) = store.prepare_merge_if_ready(1).await.unwrap().unwrap();
    first.use_append_only();
    stale.use_append_only();
    store.stage_prepared_merge(&mut first).await.unwrap();
    store.put(vec![batch("later", 2)]).await.unwrap();
    store.flush().await.unwrap();
    assert_eq!(
        store
            .commit_prepared_merge(&manifests, &manifest, first)
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store
            .commit_prepared_merge(&manifests, &manifest, stale)
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        1
    );
    assert_eq!(merge(&mut store, true).await.unwrap(), 1);
    assert_eq!(
        Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        2
    );
    store.close().await.unwrap();
}

#[derive(Debug)]
struct FailDrain;

impl CommitAuthorizer for FailDrain {
    fn authorize<'a>(
        &'a self,
        resource: &'a str,
        _: u64,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if resource.starts_with("shard:") {
                Err(Error::io("injected drain failure"))
            } else {
                Ok(())
            }
        })
    }
}

#[tokio::test]
async fn append_recovers_after_base_commit_without_reinserting_rows() {
    let dir = TempDir::new().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut store = open(uri, "a").await;
    store.put(vec![batch("event", 1)]).await.unwrap();
    store.flush().await.unwrap();
    let scope = MergeWriteScope::with_authorizer(Arc::new(FailDrain));
    let error = scope.run(merge(&mut store, true)).await.unwrap_err();
    scope.drain().await;
    assert!(
        error.to_string().contains("injected drain failure"),
        "{error}"
    );
    assert_eq!(
        Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        1
    );
    store.close().await.unwrap();
    let mut restarted = open(uri, "a").await;
    assert_eq!(merge(&mut restarted, true).await.unwrap(), 0);
    assert_eq!(
        Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        1
    );
    assert!(restarted.prepare_merge_if_ready(1).await.unwrap().is_none());
    restarted.close().await.unwrap();
}

struct PublishPeer(tokio::sync::Mutex<Option<StorageBase>>);

impl std::fmt::Debug for PublishPeer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PublishPeer")
    }
}

impl CommitAuthorizer for PublishPeer {
    fn authorize<'a>(
        &'a self,
        resource: &'a str,
        _: u64,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if resource == "base" {
                let peer = self.0.lock().await.take();
                if let Some(mut peer) = peer {
                    merge(&mut peer, true).await?;
                    peer.close().await?;
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn append_rebases_other_shards_but_replans_changed_own_coverage() {
    for same_shard in [false, true] {
        let dir = TempDir::new().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut store = open(uri, "a").await;
        store.put(vec![batch("first", 1)]).await.unwrap();
        store.flush().await.unwrap();
        let (manifests, manifest, mut prepared) =
            store.prepare_merge_if_ready(1).await.unwrap().unwrap();
        prepared.use_append_only();
        let peer = open(uri, if same_shard { "a" } else { "b" }).await;
        if !same_shard {
            peer.put(vec![batch("peer", 2)]).await.unwrap();
            peer.flush().await.unwrap();
        }
        let scope = MergeWriteScope::with_authorizer(Arc::new(PublishPeer(
            tokio::sync::Mutex::new(Some(peer)),
        )));
        let result = scope
            .run(store.commit_prepared_merge(&manifests, &manifest, prepared))
            .await;
        scope.drain().await;
        if same_shard {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("shard coverage changed during immutable append"));
        } else {
            assert_eq!(result.unwrap(), 1);
        }
        assert_eq!(
            Dataset::open(uri)
                .await
                .unwrap()
                .count_rows(None)
                .await
                .unwrap(),
            if same_shard { 1 } else { 2 }
        );
        store.close().await.unwrap();
    }
}
