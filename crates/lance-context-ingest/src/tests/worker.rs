use super::*;
use crate::{
    materializer::{self as publication, Storage},
    worker::{Assignment, Claims, Options, Worker},
};
use futures::TryStreamExt;
use std::{collections::HashMap, path::Path};

async fn fixture() -> Result<(etcd_client::Client, Claims, String)> {
    let endpoints = std::env::var("ETCD_TEST_ENDPOINTS").context("ETCD_TEST_ENDPOINTS required")?;
    let client =
        etcd_client::Client::connect(endpoints.split(',').collect::<Vec<_>>(), None).await?;
    let prefix = format!("/lance-context-test/ingest-worker/{}", Uuid::new_v4());
    let claims = Claims::new(client.clone(), prefix.clone(), 10, Duration::from_secs(3))?;
    Ok((client, claims, prefix))
}
async fn storage(path: &Path) -> Result<Storage> {
    let dataset = Dataset::open(path.to_str().context("test URI")?).await?;
    let base = dataset
        .manifest_location()
        .path
        .as_ref()
        .rsplit_once("/_versions/")
        .context("manifest base")?
        .0
        .to_owned();
    Ok(Storage {
        uri: path.to_str().unwrap().into(),
        base: object_store::path::Path::parse(base)?,
        options: HashMap::new(),
        session: Arc::new(lance::session::Session::default()),
    })
}
fn options() -> Options {
    Options {
        parallel_tables: 1,
        publications_per_claim: 4,
        poll_interval: Duration::from_millis(50),
        publication: publication::Options {
            max_source_bytes: 1 << 20,
            max_records: 8,
            retained_manifest_versions: 2,
            ..Default::default()
        },
        #[cfg(feature = "service")]
        preparers: None,
    }
}
async fn append(
    ingress: &Ingress,
    producer: &Producer,
    sequence: u64,
    id: &str,
) -> Result<(Bytes, AcceptedAppend)> {
    let session = session(0);
    let body = ipc_file(&[batch(&[id], &[&session])?])?;
    let prepared = ingress
        .prepare(
            ingress.reserve_upload(body.len()).await?,
            body.clone(),
            producer,
            sequence,
        )
        .await?;
    Ok((body, ingress.append(&prepared).await?))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ETCD_TEST_ENDPOINTS; executed by the etcd CI suite"]
async fn worker_releases_claims_without_losing_progress_past_other_tables() -> Result<()> {
    let (mut etcd, claims, prefix) = fixture().await?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    let path = root.path().join("target.lance");
    bootstrap(&c, &path, &binding()).await?;
    let target = connect(&c, binding()).await?;
    let target_storage = storage(&path).await?;
    let mut foreign = binding();
    foreign.table = Uuid::from_u128(71);
    foreign.incarnation = Uuid::from_u128(72);
    foreign.destination.dataset = "other.lance".into();
    foreign.qualified_table = "other".into();
    let other_path = root.path().join("other.lance");
    bootstrap(&c, &other_path, &foreign).await?;
    let other = connect(&c, foreign.clone()).await?;
    let open = |slot| ProducerOpen {
        bucket: 0,
        slot,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let producer = other.open_producer(open(0)).await?;
    let session = (0..1000)
        .map(|n| format!("other-session-{n}"))
        .find(|s| {
            layout()
                .session_bucket(&foreign.tenant, &foreign.qualified_table, s)
                .unwrap()
                == 0
        })
        .context("foreign bucket session")?;
    // More retained records than one scan page. No foreign publisher runs, so
    // their metadata remains in the shared source group on every target claim.
    let body = ipc_file(&[batch(&["foreign-row"], &[&session])?])?;
    for sequence in 1..=129 {
        let prepared = other
            .prepare(
                other.reserve_upload(body.len()).await?,
                body.clone(),
                &producer,
                sequence,
            )
            .await?;
        other.append(&prepared).await?;
    }
    let producer = target.open_producer(open(1)).await?;
    let (body, receipt) = append(&target, &producer, 1, "target-row").await?;
    let expected = ipc::decode(body, Arc::new(rollout_schema()))?;
    let budget = MergeMemoryBudget::new(32 << 20);
    let mut limits = options();
    limits.publication.scan_pages_per_group = 1;
    limits.publications_per_claim = 1;
    let worker = Worker::start(
        claims,
        vec![Assignment {
            ingress: target.clone(),
            storage: target_storage.clone(),
        }],
        limits,
        budget.clone(),
    )?;
    let published = target
        .wait_published(&receipt, Duration::from_secs(10))
        .await;
    let observation = worker.observation();
    worker.shutdown().await?;
    assert_eq!(budget.reserved(), 0);
    // Collect the fixture even for the negative control's publication timeout.
    let dataset = publication::read_dataset(&target, &target_storage, None).await?;
    let actual = dataset
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let other_storage = storage(&other_path).await?;
    let foreign_rows = publication::read_dataset(&other, &other_storage, None)
        .await?
        .count_rows(None)
        .await?;
    etcd.delete(
        prefix,
        Some(etcd_client::DeleteOptions::new().with_prefix()),
    )
    .await?;
    drop((dataset, target, other));
    c.shutdown().await?;
    assert!(
        observation.claims >= 2,
        "test must cross a scheduling claim"
    );
    assert_eq!(foreign_rows, 0, "target must not publish another table");
    assert_eq!(
        published.context("worker repeatedly rescanned the foreign prefix")?,
        2
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ETCD_TEST_ENDPOINTS; executed by the etcd CI suite"]
async fn expired_claim_cannot_steal_successor_epoch_and_workers_resume_prepared_lance() -> Result<()>
{
    let (mut etcd, claims, prefix) = fixture().await?;
    let root = tempfile::tempdir()?;
    let path = root.path().join("rollouts.lance");
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&c, &path, &binding()).await?;
    let ingress = connect(&c, binding()).await?;
    let storage = storage(&path).await?;
    let assignment = Assignment {
        ingress: ingress.clone(),
        storage: storage.clone(),
    };
    let budget = MergeMemoryBudget::new(32 << 20);
    let producer = ingress
        .open_producer(ProducerOpen {
            bucket: 0,
            slot: 0,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .await?;
    let (first, receipt) = append(&ingress, &producer, 1, "before-takeover").await?;
    let mut expected = BTreeMap::new();
    materializer::remember(&mut expected, first)?;
    let old = claims
        .take(&ingress)
        .await?
        .context("first claim unavailable")?;
    assert!(
        claims.take(&ingress).await?.is_none(),
        "overlapping table claim granted"
    );
    let mut old_publisher = old
        .acquire(assignment.clone(), options().publication, budget.clone())
        .await?
        .context("first claim invalid")?;
    let prepared = old_publisher
        .prepare_next()
        .await?
        .context("prepared result missing")?;
    let old_epoch = old_publisher.authority().epoch;
    let key = format!("{prefix}/{}/{}", layout().cluster, binding().table);
    let lease = etcd.get(key, None).await?.kvs()[0].lease();
    etcd.lease_revoke(lease).await?;
    let successor = claims
        .take(&ingress)
        .await?
        .context("successor claim unavailable")?;
    let mut publisher = successor
        .acquire(assignment.clone(), options().publication, budget.clone())
        .await?
        .context("successor claim invalid")?;
    assert_eq!(publisher.authority().epoch, old_epoch + 1);
    // This deliberately calls the OLD claim after the successor has committed
    // its newer epoch. It must fail the lease check and leave that epoch intact.
    assert!(old
        .acquire(assignment.clone(), options().publication, budget.clone())
        .await?
        .is_none());
    assert_eq!(publisher.state().await?.authority, *publisher.authority());
    assert!(old_publisher.run_once().await.is_err());
    assert_eq!(publisher.prepare_next().await?.unwrap(), prepared);
    let _ = old.release().await;
    assert!(
        successor.valid().await?,
        "old lease cleanup removed successor"
    );
    assert!(publisher.run_once().await?.is_some());
    assert_eq!(
        ingress
            .wait_published(&receipt, Duration::from_secs(5))
            .await?,
        2
    );
    materializer::verify(&publisher, &expected, 2).await?;
    successor.release().await?;
    drop((old_publisher, publisher));

    let (second, receipt) = append(&ingress, &producer, 2, "automatic-worker").await?;
    materializer::remember(&mut expected, second)?;
    let workers = [
        Worker::start(
            claims.clone(),
            vec![assignment.clone()],
            options(),
            budget.clone(),
        )?,
        Worker::start(claims.clone(), vec![assignment], options(), budget.clone())?,
    ];
    assert_eq!(
        ingress
            .wait_published(&receipt, Duration::from_secs(15))
            .await?,
        3
    );
    assert!(workers.iter().all(|w| w.observation().active <= 1));
    for worker in workers {
        worker.shutdown().await?;
    }
    assert_eq!(budget.reserved(), 0);
    // Read independently of either stopped publisher; no acquisition for reads.
    let dataset = publication::read_dataset(&ingress, &storage, None).await?;
    assert_eq!(dataset.version().version, 3);
    let mut found = BTreeMap::new();
    for batch in dataset
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?
    {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let id = ids.value(row);
            assert!(found.insert(id.to_owned(), ()).is_none());
            assert_eq!(batch.slice(row, 1), expected[id]);
        }
    }
    assert_eq!(found.len(), expected.len());
    etcd.delete(
        prefix,
        Some(etcd_client::DeleteOptions::new().with_prefix()),
    )
    .await?;
    drop((dataset, ingress));
    c.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires ETCD_TEST_ENDPOINTS; executed by the etcd CI suite"]
async fn abandoned_claim_expires_without_a_detached_delete() -> Result<()> {
    let (mut etcd, claims, prefix) = fixture().await?;
    let root = tempfile::tempdir()?;
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&c, &root.path().join("rollouts.lance"), &binding()).await?;
    let ingress = connect(&c, binding()).await?;
    let first = claims.take(&ingress).await?.context("claim unavailable")?;
    drop(first);
    assert!(
        claims.take(&ingress).await?.is_none(),
        "drop deleted the claim instead of letting its lease expire"
    );
    let successor = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            if let Some(claim) = claims.take(&ingress).await? {
                return Ok::<_, anyhow::Error>(claim);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    })
    .await??;
    assert!(successor.valid().await?);
    successor.release().await?;
    etcd.delete(
        prefix,
        Some(etcd_client::DeleteOptions::new().with_prefix()),
    )
    .await?;
    drop(ingress);
    c.shutdown().await
}
