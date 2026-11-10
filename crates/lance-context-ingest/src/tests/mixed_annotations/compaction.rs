use super::*;
use futures::TryStreamExt;
use std::collections::BTreeMap;

#[cfg(feature = "http")]
mod maintenance_pressure;
#[cfg(feature = "service")]
mod narrow_metadata;
mod reclamation;
#[cfg(feature = "service")]
mod recovery;
mod unchanged;

pub(in crate::tests) fn compact_binding() -> TableBinding {
    // Keep the existing 8 MiB budgets; reduce the admitted record bound so the
    // established compaction workspace formula fits, rather than enlarging it.
    TableBinding {
        max_record_bytes: 256 << 10,
        ..binding()
    }
}

pub(in crate::tests) async fn compact_owner(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
) -> Result<(Service, service::runtime::Runtime)> {
    let state = annotation_publication::state(c).await?;
    service::runtime::Runtime::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        compact_binding(),
        narrow.clone(),
        body.clone(),
        service::runtime::Configuration {
            limits: limits(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 4,
                max_batch_wait: Duration::from_millis(20),
                retained_manifest_versions: 4,
                compaction: Some(crate::materializer::SmallFileCompaction {
                    target_file_bytes: 64 << 10,
                    min_fragments: 2,
                }),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
    )
    .await
}

async fn selected_body(c: &network::Cluster, narrow: &Storage, body: &Storage) -> Result<Dataset> {
    let selected = annotation_publication::read(c, narrow, None).await?;
    let reference = BodySnapshot::read(&selected).await?.unwrap();
    mixed::load_body(body, &reference, &binding().incarnation.to_string()).await
}

async fn ids(dataset: &Dataset) -> Result<BTreeMap<String, u64>> {
    let mut scan = dataset.scan();
    scan.project(&["id"])?.with_row_id();
    let mut stream = scan.try_into_stream().await?;
    let mut result = BTreeMap::new();
    while let Some(batch) = stream.try_next().await? {
        let keys = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ids = batch
            .column_by_name("_rowid")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(result
                .insert(keys.value(row).into(), ids.value(row))
                .is_none());
        }
    }
    Ok(result)
}

async fn fragmented(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
) -> Result<(PreparedAppend, service::Completed)> {
    fragmented_with_slot(c, narrow, body, 1).await
}

pub(in crate::tests) async fn fragmented_with_slot(
    c: &network::Cluster,
    narrow: &Storage,
    body: &Storage,
    slot: u32,
) -> Result<(PreparedAppend, service::Completed)> {
    seed_with_binding(c, narrow, body, compact_binding()).await?;
    let (a, driver) = Service::connect_mixed(
        c.client.clone(),
        annotation_publication::one_shard(),
        compact_binding(),
        narrow,
        body.clone(),
        limits(),
        MergeMemoryBudget::new(8 << 20),
        MergeMemoryBudget::new(8 << 20),
    )
    .await?;
    let (stop, task) = run(driver);
    let p = producer(&a, slot).await?;
    let mut last = None;
    for (i, id) in ["A", "B", "C"].into_iter().enumerate() {
        let input = request(
            &a,
            &p,
            i as u64 + 1,
            encoded(create(&[id], &["s"], &[i as u8 + 1])?)?,
        )
        .await?;
        let result = a.submit(&input, Duration::from_secs(10)).await?;
        pressure::publish(c, &a, narrow, body).await?;
        annotation_runtime::drained(c, &a, result.index).await?;
        last = Some((input, result));
    }
    stop.send(()).unwrap();
    task.await??;
    Ok(last.unwrap())
}

#[tokio::test]
async fn mixed_automatic_compaction_drains_both_tables_and_preserves_ids_and_updates() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(60), automatic()).await?
}

async fn automatic() -> Result<()> {
    let root = artifact_root()?;
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = storage(root.path(), "annotations.lance").await?;
    let body = storage(root.path(), "body.lance").await?;
    let (last, result) = fragmented(&c, &narrow, &body).await?;
    let original_body = selected_body(&c, &narrow, &body).await?;
    assert_eq!(original_body.fragments().len(), 3);
    let original_ids = ids(&original_body).await?;
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let held = a.get_turn("A", Duration::from_secs(10)).await?.unwrap();
    let p = producer(&a, 2).await?;
    let update = request(
        &a,
        &p,
        1,
        encoded(mixed::wire::patches(
            &patch(&["A"], Some("reviewed"), Some(0.9))?,
            &limits(),
        )?)?,
    )
    .await?;
    let outcome = a.submit(&update, Duration::from_secs(10)).await?;
    annotation_runtime::drained(&c, &a, outcome.index).await?;
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let selected = annotation_publication::read(&c, &narrow, None).await?;
            let body = selected_body(&c, &narrow, &body).await?;
            if selected.fragments().len() == 1
                && body.fragments().len() == 1
                && annotation_publication::state(&c)
                    .await?
                    .progress
                    .pending
                    .is_none()
            {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await??;
    assert_eq!(
        ids(&selected_body(&c, &narrow, &body).await?).await?,
        original_ids
    );
    assert_eq!(held.annotation.status.as_deref(), Some("pending"));
    assert_eq!(
        held.body()
            .column_by_name("binary_payload")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .value(0),
        vec![1; 16 << 10]
    );
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    check(&a, "B", "s", 2, "pending", None).await?;
    check(&a, "C", "s", 3, "pending", None).await?;
    assert_eq!(a.submit(&last, Duration::from_secs(10)).await?, result);
    assert_eq!(a.submit(&update, Duration::from_secs(10)).await?, outcome);
    stop.send(()).unwrap();
    task.await??;
    let (a, runtime) = compact_owner(&c, &narrow, &body).await?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    check(&a, "A", "s", 1, "reviewed", Some(0.9)).await?;
    assert_eq!(a.submit(&update, Duration::from_secs(10)).await?, outcome);
    stop.send(()).unwrap();
    task.await??;
    c.shutdown().await
}
