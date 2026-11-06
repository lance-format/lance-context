//! Applied reads through two ingresses while native publication and cold apply wait.
use super::*;
use arrow_array::Array;
use futures::TryStreamExt;

mod fencing;
mod multipart_recovery;
mod status_filter;
mod store;
mod upload_failure;
mod wal_capacity;

#[tokio::test]
async fn owned_reads_remain_visible_during_cold_apply_and_old_publication() -> Result<()> {
    let mut root = tempfile::Builder::new();
    root.prefix("annotation-owned-cold-read-");
    let root = if let Ok(parent) = std::env::var("INGEST_TEST_ARTIFACTS") {
        root.disable_cleanup(true).tempdir_in(parent)?
    } else {
        root.tempdir()?
    };
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let gates = Arc::new(store::Gates::default());
    let storage = store::storage(&root.path().join("turns.lance"), gates.clone()).await?;
    let limits = Limits {
        retained_bytes: 256 << 10,
        retained_entries: 128,
        value_bytes: 4096,
        readers: 16,
    };
    annotation_publication::seed_contract(
        &c,
        &storage,
        annotations::Service::contract(&binding(), &limits)?,
    )
    .await?;
    let (service, runtime) = annotation_runtime::owner(&c, &storage, &limits).await?;
    let owner = Owner::new(&runtime, keys()?, options())?;
    let (stop, task) = annotation_runtime::run_owner(runtime);
    let server = Server::start(owner.router(), None).await?;
    let (ga, a) = frontend(&c, &limits, &server.uri()).await?;
    let (gb, b) = frontend(&c, &limits, &server.uri()).await?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let producer = open(&client, &a, 1).await?;

    // Block the first native data write, leaving this ACK in frozen staging.
    let mut publication = gates.pause_put();
    let first = super::super::annotations::wire(&limits, &["T"], 1, Some("reviewed"), 0, None)?;
    let _: MutationReply = update(&client, &a, &producer, 1, first.len())
        .body(first)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    publication.entered().await?;
    assert_eq!(service.observation().await.0.frozen_entries, 1);
    let frozen = point(&client, &b).await?.unwrap();
    assert_eq!(frozen.status.as_deref(), Some("reviewed"));
    assert_eq!(frozen.score, None);

    // A later field update must merge with frozen, becoming a complete active value.
    let second = super::super::annotations::wire(&limits, &["T"], 0, None, 1, Some(0.7))?;
    let _: MutationReply = update(&client, &b, &producer, 2, second.len())
        .body(second)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let active = point(&client, &a).await?.unwrap();
    assert_eq!(active.status.as_deref(), Some("reviewed"));
    assert_eq!(active.score, Some(0.7));

    // This batch touches active T and cold U. Wait for an actual data GET,
    // then require both ingress reads to finish before allowing it to return.
    let mut cold = gates.pause_get();
    let third =
        super::super::annotations::wire(&limits, &["T", "U"], 1, Some("latest"), 1, Some(0.9))?;
    let pending = tokio::spawn(
        update(&client, &a, &producer, 3, third.len())
            .body(third.clone())
            .send(),
    );
    let available = async {
        cold.entered().await?;
        ensure!(!pending.is_finished(), "cold apply received an early ACK");
        let values = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::try_join!(point(&client, &a), point(&client, &b))
        })
        .await
        .context("owned reads blocked behind actual cold apply IO")??;
        ensure!(values.0 == Some(active.clone()) && values.1 == Some(active));
        Ok::<_, anyhow::Error>(())
    }
    .await;
    cold.release();
    let response = pending.await;
    // Release both gates and join the request even when the read assertion fails.
    publication.release();
    let completed: MutationReply = response??.error_for_status()?.json().await?;
    ensure!(matches!(
        completed.completion.result,
        annotations::MutationResult::Applied
    ));

    // Old publication is now free to complete. All reads must retain the later
    // fields until the active suffix is published and staging/source drain.
    let consistent = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let (left, right) = tokio::try_join!(point(&client, &a), point(&client, &b))?;
            for value in [left, right] {
                let value = value.context("acknowledged turn disappeared")?;
                ensure!(value.status.as_deref() == Some("latest"));
                ensure!(value.score == Some(0.9));
            }
            let (view, pressured) = service.observation().await;
            if view.materialized >= completed.completion.index
                && view.retained_bytes == 0
                && view.retained_metadata_bytes == 0
                && service.source_observation().records == 0
                && !pressured
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await;
    // Both ingresses must still serve the just-published value after staging
    // drains, even when the next real data GET cannot complete. The bounded
    // published-value cache already contains this complete narrow annotation.
    let blocked = gates.pause_get();
    let published_reads = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::try_join!(point(&client, &a), point(&client, &b))
    })
    .await;
    blocked.release();
    let retried: MutationReply = update(&client, &b, &producer, 3, third.len())
        .body(third)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(retried.completion, completed.completion);

    let base = annotation_publication::read(&c, &storage, None).await?;
    let batches = base
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    let mut seen = std::collections::BTreeSet::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        let labels = batch
            .column_by_name("labels")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::ListArray>()
            .unwrap();
        let statuses = batch
            .column_by_name("status")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        let scores = batch
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .unwrap();
        let bodies = batch
            .column_by_name("body")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            assert!(seen.insert(ids.value(row).to_owned()));
            assert!(!statuses.is_null(row) && !scores.is_null(row) && !labels.is_null(row));
            assert_eq!(statuses.value(row), "latest");
            assert_eq!(scores.value(row), 0.9);
            let label_values = labels.value(row);
            let label_values = label_values
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap();
            assert_eq!(
                label_values.iter().collect::<Vec<_>>(),
                vec![Some("original")]
            );
            let body = bodies.value(row);
            assert_eq!(body.len(), 256 << 10);
            assert!(body
                .iter()
                .enumerate()
                .all(|(i, b)| *b == ((i * 31 + i / 17) % 251) as u8));
        }
    }
    assert_eq!(seen, ["T".to_owned(), "U".to_owned()].into());
    stop.send(()).unwrap();
    task.await??;
    ga.shutdown().await;
    gb.shutdown().await;
    a.shutdown().await?;
    b.shutdown().await?;
    owner.shutdown().await;
    server.shutdown().await?;
    c.shutdown().await?;
    available?;
    consistent??;
    let published_reads =
        published_reads.context("published cross-ingress reads waited for blocked data IO")??;
    for value in [published_reads.0, published_reads.1] {
        let value = value.context("published annotation disappeared")?;
        ensure!(value.status.as_deref() == Some("latest"));
        ensure!(value.score == Some(0.9));
    }
    Ok(())
}
