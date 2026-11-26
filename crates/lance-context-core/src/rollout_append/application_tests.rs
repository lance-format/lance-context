use super::*;
use crate::merge_write_scope::{CommitAuthorizer, MergeWriteScope};
use crate::StorageBaseOptions;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::Field;
use std::{future::Future, pin::Pin};

fn batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("event_id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["event"])),
            Arc::new(Int64Array::from(vec![7])),
        ],
    )
    .unwrap()
}

async fn writer(uri: &str, shard: &str) -> StorageBase {
    let mut options = StorageBaseOptions::new(batch().schema(), "event_id");
    options.shard_id = Some(shard.to_owned());
    options.seal_on_put = false;
    StorageBase::open(uri, options, true).await.unwrap()
}

fn options(policy: ReplayPolicy) -> AppendOptions {
    let mut options = AppendOptions::new(batch().schema(), "event_id");
    options.replay_policy = policy;
    options
}

async fn parts(
    uri: &str,
    coordinator: &mut AppendCoordinator,
    policy: ReplayPolicy,
) -> Vec<StagedAppend> {
    let (plans, _) = coordinator.plan(&[], 64, 1 << 20).await.unwrap();
    let mut parts = Vec::new();
    for plan in plans {
        parts.push(
            stage_with_options(uri, plan, MergeMemoryBudget::new(8 << 20), options(policy))
                .await
                .unwrap(),
        );
    }
    parts
}

#[tokio::test]
async fn application_schema_and_key_share_rollout_staging_with_explicit_replay_policy() {
    for policy in [
        ReplayPolicy::CheckBaseOnRecovery,
        ReplayPolicy::AllowDuplicates,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut a = writer(uri, "a").await;
        a.put(vec![batch()]).await.unwrap();
        a.flush().await.unwrap();
        let mut coordinator = AppendCoordinator::open_with_options(uri, options(policy))
            .await
            .unwrap();
        let first = parts(uri, &mut coordinator, policy).await;
        coordinator.commit(first).await.unwrap();
        // A newly discovered shard retries the same immutable identity. The
        // rollout-compatible policy checks base; application-owned read dedup opts out.
        let mut b = writer(uri, "b").await;
        b.put(vec![batch()]).await.unwrap();
        b.flush().await.unwrap();
        let next = parts(uri, &mut coordinator, policy).await;
        coordinator.commit(next).await.unwrap();
        assert_eq!(
            Dataset::open(uri)
                .await
                .unwrap()
                .count_rows(None)
                .await
                .unwrap(),
            if policy == ReplayPolicy::CheckBaseOnRecovery {
                1
            } else {
                2
            }
        );
        assert!(coordinator
            .plan(&[], 64, 1 << 20)
            .await
            .unwrap()
            .0
            .is_empty());
        assert!(
            AppendCoordinator::open(uri, None).await.is_err(),
            "rollout wrapper must still reject another schema"
        );
        a.close().await.unwrap();
        b.close().await.unwrap();
    }
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
async fn application_append_recovers_commit_before_drain_without_reinserting() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut writer = writer(uri, "a").await;
    writer.put(vec![batch()]).await.unwrap();
    writer.flush().await.unwrap();
    let scope = MergeWriteScope::with_authorizer(Arc::new(FailDrain));
    let staged = scope
        .run(async {
            let mut coordinator =
                AppendCoordinator::open_with_options(uri, options(ReplayPolicy::AllowDuplicates))
                    .await
                    .unwrap();
            let staged = parts(uri, &mut coordinator, ReplayPolicy::AllowDuplicates).await;
            let error = coordinator.commit(staged.clone()).await.unwrap_err();
            assert!(
                error.to_string().contains("injected drain failure"),
                "{error}"
            );
            staged
        })
        .await;
    scope.drain().await;
    assert_eq!(
        Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        1
    );
    let mut successor =
        AppendCoordinator::open_with_options(uri, options(ReplayPolicy::AllowDuplicates))
            .await
            .unwrap();
    assert_eq!(successor.commit(staged).await.unwrap(), 1);
    assert_eq!(
        Dataset::open(uri)
            .await
            .unwrap()
            .count_rows(None)
            .await
            .unwrap(),
        1
    );
    assert!(successor.plan(&[], 64, 1 << 20).await.unwrap().0.is_empty());
    writer.close().await.unwrap();
}
