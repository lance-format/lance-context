use super::*;
use crate::materializer::{Acquisition, Options, Publisher, Storage};
use futures::TryStreamExt;
use object_store::path::Path as ObjectPath;
use std::collections::HashMap;

pub(super) async fn publisher(
    ingress: &Ingress,
    storage: &Storage,
    budget: &Arc<MergeMemoryBudget>,
    epoch: u64,
) -> Result<Publisher> {
    Publisher::acquire(
        ingress.clone(),
        storage.clone(),
        Options {
            max_source_bytes: 1 << 20,
            max_records: 8,
            write_parallelism: 2,
            retained_manifest_versions: 2,
            ..Options::default()
        },
        budget.clone(),
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: epoch,
        },
    )
    .await
}
async fn append(
    ingress: &Ingress,
    open: &ProducerOpen,
    sequence: u64,
    ids: &[&str],
) -> Result<(Bytes, AcceptedAppend)> {
    let producer = ingress.open_producer(open.clone()).await?;
    let session = session(open.bucket);
    let body = ipc_file(&[batch(ids, &vec![session.as_str(); ids.len()])?])?;
    let prepared = ingress
        .prepare(
            ingress.reserve_upload(body.len()).await?,
            body.clone(),
            &producer,
            sequence,
        )
        .await?;
    let accepted = ingress.append(&prepared).await?;
    Ok((body, accepted))
}
pub(super) async fn verify(
    publisher: &Publisher,
    expected: &BTreeMap<String, RecordBatch>,
    version: u64,
) -> Result<()> {
    let dataset = publisher.read_dataset(None).await?;
    assert_eq!(dataset.version().version, version);
    let batches = dataset
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let mut found = BTreeMap::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let key = ids.value(row).to_owned();
            assert!(
                found.insert(key.clone(), ()).is_none(),
                "duplicate published id"
            );
            let actual = batch.slice(row, 1);
            let expected = &expected[&key];
            assert_eq!(actual.schema(), expected.schema());
            for (left, right) in actual.columns().iter().zip(expected.columns()) {
                assert_eq!(left, right, "column mismatch for {key}");
            }
        }
    }
    assert_eq!(
        found.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    Ok(())
}
pub(super) fn remember(expected: &mut BTreeMap<String, RecordBatch>, body: Bytes) -> Result<()> {
    for batch in ipc::decode(body, Arc::new(rollout_schema()))? {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(expected
                .insert(ids.value(row).to_owned(), batch.slice(row, 1))
                .is_none());
        }
    }
    Ok(())
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn real_rollout_files_resume_after_preparation_and_publication_restart_then_retire(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("rollouts.lance");
    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    bootstrap(&c, &path, &binding()).await?;
    let base = Dataset::open(path.to_str().unwrap()).await?;
    let object_base = base
        .manifest_location()
        .path
        .as_ref()
        .rsplit_once("/_versions/")
        .context("manifest base")?
        .0
        .to_owned();
    let storage = Storage {
        uri: path.to_str().unwrap().into(),
        base: ObjectPath::parse(object_base)?,
        options: HashMap::new(),
        session: Arc::new(lance::session::Session::default()),
    };
    drop(base);
    let budget = MergeMemoryBudget::new(32 << 20);
    let ingress = connect(&c, binding()).await?;
    let mut expected = BTreeMap::new();
    let mut originals = Vec::new();
    for bucket in 0..4 {
        let open = ProducerOpen {
            bucket,
            slot: bucket,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        };
        let ids = [format!("{bucket}-a"), format!("{bucket}-b")];
        let (body, receipt) = append(&ingress, &open, 1, &[&ids[0], &ids[1]]).await?;
        remember(&mut expected, body.clone())?;
        originals.push((open, body, receipt));
    }
    let mut first = publisher(&ingress, &storage, &budget, 1).await?;
    let saved = first
        .prepare_next()
        .await?
        .context("no staged publication")?;
    assert_eq!(
        first
            .state()
            .await?
            .progress
            .pending
            .as_ref()
            .unwrap()
            .plan
            .sources
            .len(),
        2
    );
    assert_eq!(budget.reserved(), 0);
    let old_authority = first.authority().clone();
    drop((first, ingress));
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&c, binding()).await?;
    let mut second = publisher(&ingress, &storage, &budget, 2).await?;
    assert_eq!(
        second.prepare_next().await?.unwrap(),
        saved,
        "recovery rewrote durable prepared files"
    );
    let old = c
        .client
        .request_routed(Request::Commit(Box::new(Command::Prepare(
            p::PrepareRequest {
                authority: old_authority,
                transaction: saved.transaction,
                plan_index: saved.plan_index,
                descriptors: saved.descriptors.clone(),
            },
        ))))
        .await?;
    assert!(matches!(
        old.reply,
        Reply::Commit(Outcome::Committed(Applied::Publisher(p::Reply::Rejected(
            p::Rejection::Authority
        ))))
    ));
    let published = second
        .commit_pending()
        .await?
        .context("publication missing")?;
    verify(&second, &expected, 2).await?;
    for (_, _, receipt) in &originals {
        assert_eq!(
            ingress.publication_status(receipt).await?,
            PublicationStatus::Pending
        );
    }
    // Crash boundary: the real manifest is selected and confirmed, but source
    // closure/cursor advancement has not yet run. New authority must resume it.
    drop((second, ingress));
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&c, binding()).await?;
    let mut third = publisher(&ingress, &storage, &budget, 3).await?;
    assert_eq!(third.commit_pending().await?.unwrap(), published);
    let advanced = third
        .finish_pending()
        .await?
        .context("advancement missing")?;
    assert_eq!(advanced.sources.len(), 2);
    for (open, body, receipt) in &originals {
        assert_eq!(
            ingress.publication_status(receipt).await?,
            PublicationStatus::Published {
                minimum_read_version: 2
            }
        );
        let producer = ingress.open_producer(open.clone()).await?;
        let exact = ingress
            .prepare(
                ingress.reserve_upload(body.len()).await?,
                body.clone(),
                &producer,
                1,
            )
            .await?;
        assert_eq!(ingress.append(&exact).await?.position(), receipt.position());
        for node in 1..=3 {
            tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    let response = c
                        .client
                        .request(
                            node,
                            Request::Fetch {
                                group: receipt.position().group,
                                index: receipt.position().index,
                            },
                        )
                        .await?;
                    if matches!(response.reply, Reply::Record(None)) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Ok::<_, anyhow::Error>(())
            })
            .await??;
        }
    }
    assert!(third.run_once().await?.is_none());
    verify(&third, &expected, 2).await?;
    for sequence in 2..=5 {
        for (open, _, _) in &originals {
            let id = format!("{}-round-{sequence}", open.bucket);
            let (body, _) = append(&ingress, open, sequence, &[&id]).await?;
            remember(&mut expected, body)?;
        }
        assert!(third.run_once().await?.is_some());
        verify(&third, &expected, sequence + 1).await?;
        assert!(
            third.state().await?.manifests.unwrap().versions.len() <= 3,
            "catalog history grew beyond policy"
        );
    }
    assert_eq!(budget.reserved(), 0);
    drop((third, ingress));
    c.shutdown().await?;

    let c = network::Cluster::open_signed(root.path(), layout()).await?;
    let ingress = connect(&c, binding()).await?;
    let mut final_owner = publisher(&ingress, &storage, &budget, 4).await?;
    assert!(final_owner.run_once().await?.is_none());
    verify(&final_owner, &expected, 6).await?;
    assert!(
        final_owner.read_dataset(Some(2)).await.is_err(),
        "retired history unexpectedly resolved"
    );
    drop(final_owner);
    c.shutdown().await
}
