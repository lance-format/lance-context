use super::*;
use crate::{
    bootstrap::annotations::{self as handoff, Snapshot},
    tests::{annotation_publication, annotation_runtime, annotations as requests},
};
use arrow_array::{Float32Array, LargeStringArray};
use futures::TryStreamExt;
use lance::dataset::CommitBuilder;
use lance_context_core::annotation_staging::Limits;

fn limits() -> Limits {
    Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    }
}

pub(crate) async fn original(storage: &Storage) -> Result<Dataset> {
    let original = batch(&["T", "U"], &["s", "s"])?;
    let mut columns = original.columns().to_vec();
    columns[original.schema().index_of("score")?] =
        Arc::new(Float32Array::from(vec![Some(0.5); 2]));
    columns[original.schema().index_of("content")?] =
        Arc::new(LargeStringArray::from(vec!["body".repeat(64 << 10); 2]));
    let batch = RecordBatch::try_new(original.schema(), columns)?;
    Ok(Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], original.schema()),
        &storage.uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            ..Default::default()
        }),
    )
    .await?)
}

#[tokio::test]
async fn annotation_handoff_preserves_real_rollout_data_and_resumes_after_schema_commit(
) -> Result<()> {
    for interrupted in [false, true] {
        let root = tempfile::tempdir()?;
        let storage = Storage::resolve(
            root.path().join("rollouts.lance").to_str().unwrap().into(),
            Default::default(),
            Arc::default(),
        )
        .await?;
        let base = original(&storage).await?;
        let files = base.manifest().fragments.clone();
        let snapshot = Snapshot {
            version: base.version_id(),
            transaction: Uuid::parse_str(&base.read_transaction().await?.unwrap().uuid)?,
        };
        let layout = annotation_publication::one_shard();
        let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
        let identity = identity();
        if interrupted {
            let binding = binding();
            let scope = Scope {
                cluster: layout.cluster,
                layout: layout.fingerprint()?,
                table: binding.table,
                destination: binding.destination.clone(),
                incarnation: binding.incarnation,
                contract: crate::annotations::Service::contract(&binding, &limits())?,
            };
            let response = c
                .client
                .request_routed(Request::Commit(Box::new(Command::Acquire(p::Acquire {
                    table: scope.table,
                    destination: scope.destination.clone(),
                    ingest: scope.contract.clone(),
                    owner: identity.owner,
                    request: identity.acquisition,
                    expected_epoch: 0,
                }))))
                .await?;
            assert!(matches!(
                response.reply,
                Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Acquired(
                    _
                ))))
            ));
            let marker = serde_json::to_string(&(&scope, &identity, &snapshot))?;
            let transaction = handoff::prepare(&base, &scope, &identity, &marker)?;
            let selected = CommitBuilder::new(Arc::new(base.clone()))
                .with_commit_handler(Arc::new(handoff::ExactVersion(snapshot.version + 1)))
                .with_max_retries(0)
                .with_skip_auto_cleanup(true)
                .execute(transaction)
                .await?;
            assert_eq!(selected.manifest().fragments, files);
            assert!(view(&c, binding.table).await?.manifests.is_none());
        }
        handoff::provision(
            c.client.clone(),
            layout.clone(),
            binding(),
            storage.clone(),
            identity.clone(),
            snapshot.clone(),
            limits(),
        )
        .await?;
        let initialized = view(&c, binding().table).await?;
        assert_eq!(initialized.authority.epoch, 1);
        assert_eq!(initialized.manifests.as_ref().unwrap().initial_version, 2);
        let migrated = Dataset::open(&storage.uri).await?;
        assert_eq!(migrated.manifest().fragments, files);
        assert_eq!(migrated.version_id(), 2);
        // Every original column, including body and preexisting score, still
        // references the same data files and returns the same Arrow values.
        let names = base
            .schema()
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>();
        let expected = base
            .scan()
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let actual = migrated
            .scan()
            .project(&names)?
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.columns(), expected.columns());
        }
        let (service, runtime) = annotation_runtime::owner(&c, &storage, &limits()).await?;
        let (stop, task) = annotation_runtime::run_owner(runtime);
        let initial = service.get("T", Duration::from_secs(30)).await?.unwrap();
        assert!(initial.labels.is_empty());
        assert_eq!(initial.status, None);
        assert_eq!(initial.score, Some(0.5));
        let producer = requests::producer(&service, 1).await?;
        for sequence in 1..=2 {
            let bytes = requests::wire(&limits(), &["T"], 1, Some("reviewed"), 0, None)?;
            let request = requests::request(&service, &producer, sequence, bytes).await?;
            let receipt = service.submit(&request, Duration::from_secs(30)).await?;
            if sequence == 2 {
                annotation_runtime::drained(&service, receipt.index).await?;
            }
        }
        let current = service.get("T", Duration::from_secs(30)).await?.unwrap();
        assert_eq!(current.status.as_deref(), Some("reviewed"));
        assert_eq!(current.score, Some(0.5));
        assert!(current.labels.is_empty());
        let untouched = service.get("U", Duration::from_secs(30)).await?.unwrap();
        assert!(untouched.labels.is_empty());
        assert_eq!(untouched.status, None);
        assert_eq!(untouched.score, Some(0.5));
        let before_retry = view(&c, binding().table).await?;
        handoff::provision(
            c.client.clone(),
            layout.clone(),
            binding(),
            storage.clone(),
            identity.clone(),
            snapshot.clone(),
            limits(),
        )
        .await?;
        assert_eq!(view(&c, binding().table).await?, before_retry);
        let mut wrong = snapshot.clone();
        wrong.transaction = Uuid::new_v4();
        assert!(handoff::provision(
            c.client.clone(),
            layout.clone(),
            binding(),
            storage.clone(),
            identity.clone(),
            wrong,
            limits()
        )
        .await
        .is_err());
        assert_eq!(view(&c, binding().table).await?, before_retry);
        stop.send(()).unwrap();
        task.await??;
        c.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn annotation_handoff_refuses_changed_snapshot_and_existing_wal_policy() -> Result<()> {
    let root = tempfile::tempdir()?;
    let storage = Storage::resolve(
        root.path().join("rollouts.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let base = original(&storage).await?;
    let snapshot = Snapshot {
        version: 1,
        transaction: Uuid::new_v4(),
    };
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let mut invalid_limits = limits();
    invalid_limits.retained_entries = 0;
    assert!(handoff::provision(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        identity(),
        Snapshot {
            version: 1,
            transaction: Uuid::parse_str(&base.read_transaction().await?.unwrap().uuid)?,
        },
        invalid_limits,
    )
    .await
    .is_err());
    let response = c
        .client
        .request_routed(Request::Query {
            group: c.client.catalog_group(),
            query: query::Query::Publisher {
                table: binding().table,
            },
        })
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Query(query::Outcome::Observed {
            value: query::Value::Publisher(None),
            ..
        })
    ));
    assert!(handoff::provision(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        identity(),
        snapshot,
        limits()
    )
    .await
    .is_err());
    assert_eq!(Dataset::open(&storage.uri).await?.version_id(), 1);
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Acquire(p::Acquire {
            table: binding().table,
            destination: binding().destination,
            ingest: binding().contract()?,
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: 0,
        }))))
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Acquired(
            _
        ))))
    ));
    let snapshot = Snapshot {
        version: 1,
        transaction: Uuid::parse_str(&base.read_transaction().await?.unwrap().uuid)?,
    };
    assert!(handoff::provision(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        storage.clone(),
        identity(),
        snapshot,
        limits()
    )
    .await
    .is_err());
    assert_eq!(Dataset::open(&storage.uri).await?.version_id(), 1);
    c.shutdown().await
}

#[tokio::test]
async fn annotation_handoff_rejects_lance_initial_rebase_before_publication() -> Result<()> {
    let root = tempfile::tempdir()?;
    let storage = Storage::resolve(
        root.path().join("rollouts.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    let base = original(&storage).await?;
    let scope = Scope {
        cluster: annotation_publication::one_shard().cluster,
        layout: annotation_publication::one_shard().fingerprint()?,
        table: binding().table,
        destination: binding().destination,
        incarnation: binding().incarnation,
        contract: crate::annotations::Service::contract(&binding(), &limits())?,
    };
    let transaction = handoff::prepare(&base, &scope, &identity(), "test-handoff")?;
    let mut current = base.clone();
    current.update_metadata([("concurrent", "commit")]).await?;
    let current = Dataset::open(&storage.uri).await?;
    let result = CommitBuilder::new(Arc::new(base))
        .with_commit_handler(Arc::new(handoff::ExactVersion(2)))
        .with_max_retries(0)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("annotation handoff cannot rebase"));
    let latest = Dataset::open(&storage.uri).await?;
    assert_eq!(latest.version_id(), 2);
    assert_eq!(latest.schema(), current.schema());
    assert_eq!(
        latest.read_transaction().await?,
        current.read_transaction().await?
    );
    Ok(())
}
