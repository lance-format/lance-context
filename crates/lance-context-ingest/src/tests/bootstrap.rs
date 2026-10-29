use super::*;
use crate::{
    bootstrap::{provision, Identity},
    materializer::Storage,
};
use arrow_array::RecordBatchIterator;
use lance::dataset::WriteParams;
use lance_table::io::commit::ConditionalPutCommitHandler;
use std::collections::HashMap;

mod annotation_projection;
pub(super) mod annotations;

pub(super) fn identity() -> Identity {
    Identity {
        owner: Uuid::from_u128(9000),
        acquisition: Uuid::from_u128(9001),
        manifest: Uuid::from_u128(9002),
    }
}
async fn view(c: &network::Cluster, table: Uuid) -> Result<Box<p::Table>> {
    let response = c
        .client
        .request_routed(Request::Query {
            group: c.client.catalog_group(),
            query: query::Query::Publisher { table },
        })
        .await?;
    let Reply::Query(query::Outcome::Observed {
        value: query::Value::Publisher(Some(view)),
        ..
    }) = response.reply
    else {
        bail!("missing publisher state");
    };
    Ok(view)
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn production_bootstrap_restarts_idempotently_and_refuses_foreign_empty_genesis() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let path = root.path().join("managed.lance");
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    provision(
        c.client.clone(),
        layout(),
        binding(),
        storage.clone(),
        identity(),
    )
    .await?;
    let original = view(&c, binding().table).await?;
    assert_eq!(original.authority.epoch, 1);
    assert_eq!(original.manifests.as_ref().unwrap().head, 1);
    assert!(Dataset::open(&storage.uri)
        .await?
        .manifest()
        .uses_stable_row_ids());
    provision(
        c.client.clone(),
        layout(),
        binding(),
        storage.clone(),
        identity(),
    )
    .await?;
    assert_eq!(view(&c, binding().table).await?, original);
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    provision(
        c.client.clone(),
        layout(),
        binding(),
        storage.clone(),
        identity(),
    )
    .await?;
    assert_eq!(view(&c, binding().table).await?, original);
    let mut foreign_identity = identity();
    foreign_identity.manifest = Uuid::new_v4();
    assert!(provision(
        c.client.clone(),
        layout(),
        binding(),
        storage,
        foreign_identity
    )
    .await
    .is_err());
    assert_eq!(view(&c, binding().table).await?, original);

    let path = root.path().join("foreign.lance");
    let foreign = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch(&[], &[])?)], Arc::new(rollout_schema())),
        &foreign.uri,
        Some(WriteParams {
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            ..Default::default()
        }),
    )
    .await?;
    let mut other = binding();
    other.table = Uuid::new_v4();
    other.destination.dataset = "foreign.lance".into();
    assert!(provision(
        c.client.clone(),
        layout(),
        other.clone(),
        foreign.clone(),
        identity()
    )
    .await
    .is_err());
    assert!(view(&c, other.table).await?.manifests.is_none());
    let untouched = Dataset::open(&foreign.uri).await?;
    assert_eq!(untouched.version().version, 1);
    assert_eq!(untouched.count_rows(None).await?, 0);
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_recovers_legacy_genesis_without_changing_row_ids() -> Result<()> {
    let root = tempfile::tempdir()?;
    let storage = Storage::resolve(
        root.path().join("legacy.lance").to_str().unwrap().into(),
        Default::default(),
        Arc::default(),
    )
    .await?;
    // An older binary wrote the original genesis and stopped before selecting
    // it in the WAL catalog. Preserve the exact persisted bootstrap marker.
    let marker = serde_json::to_string(&(scope(&binding()), identity()))?;
    let original = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch(&[], &[])?)], Arc::new(rollout_schema())),
        &storage.uri,
        Some(WriteParams {
            enable_stable_row_ids: false,
            commit_handler: Some(Arc::new(ConditionalPutCommitHandler)),
            transaction_properties: Some(Arc::new(HashMap::from([(
                "lance-context.ingest.genesis.v1".to_owned(),
                marker,
            )]))),
            ..Default::default()
        }),
    )
    .await?;
    assert!(!original.manifest().uses_stable_row_ids());
    let transaction = original.read_transaction_by_version(1).await?.unwrap();
    let location = original.manifest_location().clone();
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    for _ in 0..2 {
        provision(
            c.client.clone(),
            layout(),
            binding(),
            storage.clone(),
            identity(),
        )
        .await?;
        let selected = view(&c, binding().table).await?;
        let manifests = selected.manifests.as_ref().unwrap();
        assert_eq!(manifests.head, 1);
        assert_eq!(
            manifests.versions[&1].transaction.to_string(),
            transaction.uuid
        );
        let reopened = Dataset::open(&storage.uri).await?;
        assert_eq!(reopened.version().version, 1);
        assert_eq!(reopened.manifest_location().path, location.path);
        assert!(!reopened.manifest().uses_stable_row_ids());
    }
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_reuses_original_genesis_after_restart_before_catalog_selection() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("interrupted.lance");
    let storage = Storage::resolve(
        path.to_str().unwrap().into(),
        Default::default(),
        Arc::new(lance::session::Session::default()),
    )
    .await?;
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    let b = binding();
    let id = identity();
    let response = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Acquire(p::Acquire {
            table: b.table,
            destination: b.destination.clone(),
            ingest: b.contract()?,
            owner: id.owner,
            request: id.acquisition,
            expected_epoch: 0,
        }))))
        .await?;
    assert!(matches!(
        response.reply,
        Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Acquired(
            _
        ))))
    ));
    let scope = Scope {
        cluster: layout().cluster,
        layout: layout().fingerprint()?,
        table: b.table,
        destination: b.destination.clone(),
        incarnation: b.incarnation,
        contract: b.contract()?,
    };
    let (object, transaction) = crate::bootstrap::genesis(&storage, &scope, &id).await?;
    assert!(view(&c, b.table).await?.manifests.is_none());
    // The actual genesis file/transaction exists, but no manifest was selected.
    // Reopen all original replica directories and retry the persisted identity.
    c.shutdown().await?;
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    provision(c.client.clone(), layout(), b.clone(), storage, id).await?;
    let state = view(&c, b.table).await?;
    assert_eq!(state.authority.epoch, 1);
    let selected = &state.manifests.as_ref().unwrap().versions[&1];
    assert_eq!(selected.object, object);
    assert_eq!(selected.transaction, transaction);
    c.shutdown().await
}
