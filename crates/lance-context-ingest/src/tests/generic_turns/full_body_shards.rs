//! Two session shards of one logical full-body table. HTTP writes and selected
//! manifests and cross-ingress full-body reads use the real owner/forwarder RPC.
//! Three local WAL nodes host two RF3 groups; this is not a scale benchmark.
use super::http::{journal, keys, transport, Server};
use super::*;
use crate::{
    annotations::{
        runtime::{Configuration, Runtime},
        single_body,
    },
    http::{
        annotations::{Forwarder, Owner},
        journal::DurableMutationClient,
    },
    materializer::{Acquisition, Options},
    tests::annotation_runtime::{automatic_reclamation::deletion_faults::fault_store, run_owner},
};

mod read_safety;
#[cfg(feature = "streaming-producer")]
mod streaming;

struct Physical {
    binding: TableBinding,
    shard: Shard,
    storage: Storage,
}

impl Physical {
    async fn state(&self, c: &network::Cluster) -> Result<Box<p::Table>> {
        let response = c
            .client
            .request_routed(Request::Query {
                group: c.client.catalog_group(),
                query: query::Query::Publisher {
                    table: self.binding.table,
                },
            })
            .await?;
        let Reply::Query(query::Outcome::Observed {
            value: query::Value::Publisher(Some(state)),
            ..
        }) = response.reply
        else {
            bail!("missing physical body table");
        };
        Ok(state)
    }

    async fn owner(
        &self,
        c: &network::Cluster,
        layout: &Arc<Layout>,
        spec: &TurnIngestionSpec,
        uploads: Arc<MergeMemoryBudget>,
    ) -> Result<(Service, Runtime)> {
        self.owner_with_limits(c, layout, spec, uploads, fixture::limits())
            .await
    }

    async fn owner_with_limits(
        &self,
        c: &network::Cluster,
        layout: &Arc<Layout>,
        spec: &TurnIngestionSpec,
        uploads: Arc<MergeMemoryBudget>,
        limits: lance_context_core::annotation_staging::Limits,
    ) -> Result<(Service, Runtime)> {
        Runtime::connect_body_shard(
            c.client.clone(),
            layout.clone(),
            self.binding.clone(),
            self.storage.clone(),
            Configuration {
                limits,
                upload_budget: uploads,
                apply_budget: MergeMemoryBudget::new(8 << 20),
                publication_budget: MergeMemoryBudget::new(8 << 20),
                publication: Options {
                    max_source_bytes: 1 << 20,
                    min_source_bytes: 1 << 20,
                    max_records: 64,
                    max_batch_wait: Duration::from_millis(20),
                    ..Default::default()
                },
            },
            Acquisition {
                owner: Uuid::new_v4(),
                request: Uuid::new_v4(),
                expected_epoch: self.state(c).await?.authority.epoch,
            },
            self.shard,
            spec,
        )
        .await
    }

    async fn client(
        &self,
        root: &std::path::Path,
        base: reqwest::Url,
        layout: &Layout,
        spec: &TurnIngestionSpec,
        slot: u32,
    ) -> Result<DurableMutationClient> {
        std::fs::create_dir_all(root)?;
        DurableMutationClient::open_full_body(
            journal(
                root,
                base,
                ProducerOpen {
                    bucket: self.shard.bucket,
                    slot,
                    expected_generation: 0,
                    operation: Uuid::from_u128(100 + slot as u128),
                },
            ),
            layout,
            self.binding.clone(),
            spec,
            &fixture::limits(),
            self.shard,
            transport()?,
            MergeMemoryBudget::new(64 << 20),
        )
        .await
    }

    fn wire(&self, spec: &TurnIngestionSpec, body: &RecordBatch) -> Result<Bytes> {
        fixture::encoded(single_body::batch(
            spec,
            &fixture::limits(),
            self.shard,
            body,
        )?)
    }

    async fn drained(&self, c: &network::Cluster, service: &Service, cut: u64) -> Result<()> {
        let group = service
            .ingress()
            .layout
            .group_for_bucket(self.shard.bucket)?;
        tokio::time::timeout(DEADLINE, async {
            loop {
                let (view, pressured) = service.observation().await;
                let state = self.state(c).await?;
                if view.materialized >= cut
                    && view.retained_bytes == 0
                    && service.retained_body_frames() == (0, 0)
                    && service.source_observation().records == 0
                    && !pressured
                    && state.progress.pending.is_none()
                    && state.progress.checkpoint.get(&group).copied().unwrap_or(0) >= cut
                {
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?
    }
}

fn body(session: &str, id: &str, value: u8) -> Result<RecordBatch> {
    RecordBatch::try_from_iter(vec![
        (
            "session_id",
            Arc::new(StringArray::from(vec![session])) as ArrayRef,
        ),
        ("id", Arc::new(StringArray::from(vec![id])) as ArrayRef),
        (
            "content",
            Arc::new(LargeBinaryArray::from_iter_values([vec![value; 256 << 10]])) as ArrayRef,
        ),
    ])
    .map_err(Into::into)
}

async fn check(service: &Service, spec: &TurnIngestionSpec, expected: &RecordBatch) -> Result<()> {
    let key = spec.row_key(expected, 0, 4096)?;
    let value = service
        .get_full_body(&key, DEADLINE)
        .await?
        .context("acknowledged body missing")?;
    assert_eq!(value.batch(), expected);
    Ok(())
}

async fn http_check(
    base: &reqwest::Url,
    spec: &TurnIngestionSpec,
    expected: &RecordBatch,
) -> Result<()> {
    let response = transport()?
        .post(base.join(&format!(
            "/api/v1/annotations/{}/read-body",
            binding().qualified_table
        ))?)
        .json(&crate::http::annotations::PointRequest {
            turn: spec.row_key(expected, 0, 4096)?,
        })
        .send()
        .await?;
    let status = response.status();
    if status != reqwest::StatusCode::OK {
        bail!("body read returned {status}: {}", response.text().await?);
    }
    assert_eq!(
        response.headers()[reqwest::header::CONTENT_TYPE],
        "application/vnd.apache.arrow.file"
    );
    let bytes = response.bytes().await?;
    assert!(
        bytes.len() > 128 << 10,
        "exercise a body larger than narrow JSON replies"
    );
    let key = spec.row_key(expected, 0, 4096)?;
    let decoded = crate::http::annotations::full_body::decode(bytes.clone(), spec, &key)?;
    assert_eq!(&decoded, expected);
    let wrong_key = spec.business_key(&["other-session", "other-id"], 4096)?;
    assert!(crate::http::annotations::full_body::decode(bytes, spec, &wrong_key).is_err());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_body_session_shards_read_unpublished_and_drain_independently_after_restart(
) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(120), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut definition = (*layout()).clone();
    definition.buckets = vec![10, 20];
    let layout = Arc::new(definition);
    let sessions = (0..2)
        .map(|bucket| {
            (0..1000)
                .map(|i| format!("session-{i}"))
                .find(|s| Shard::for_session(&layout, &binding(), s).unwrap().bucket == bucket)
                .unwrap()
        })
        .collect::<Vec<_>>();
    // Same ID in two sessions is two distinct bodies on different shards.
    let input = [
        body(&sessions[0], "same-id", 7)?,
        body(&sessions[1], "same-id", 9)?,
    ];
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let faults = Arc::new(fault_store::Faults::default());
    let mut shards = Vec::new();
    for bucket in 0..2 {
        let mut table = binding();
        table.table = Uuid::new_v4();
        table.incarnation = Uuid::new_v4();
        table.destination.dataset = format!("full-body-{bucket}.lance");
        let storage = fault_store::storage(
            &root.path().join(&table.destination.dataset),
            if bucket == 0 {
                faults.clone()
            } else {
                Arc::default()
            },
        )
        .await?;
        let shard = Shard { bucket };
        crate::bootstrap::provision_body_shard(
            c.client.clone(),
            layout.clone(),
            table.clone(),
            storage.clone(),
            Identity {
                owner: Uuid::new_v4(),
                acquisition: Uuid::new_v4(),
                manifest: Uuid::new_v4(),
            },
            &fixture::limits(),
            shard,
            &spec,
        )
        .await?;
        shards.push(Physical {
            binding: table,
            shard,
            storage,
        });
    }
    // Both HTTP layers share this test budget. Admit one owner's scan/output
    // workspace plus the forwarding response for the 5 MiB record contract.
    let uploads = MergeMemoryBudget::new(32 << 20);
    let (a, ra) = shards[0].owner(&c, &layout, &spec, uploads.clone()).await?;
    let (b, rb) = shards[1].owner(&c, &layout, &spec, uploads.clone()).await?;
    let oa = Owner::new(&ra, keys()?, Default::default())?;
    let ob = Owner::new(&rb, keys()?, Default::default())?;
    let server_a = Server::start(oa.router()).await?;
    let server_b = Server::start(ob.router()).await?;
    let routes = vec![
        (a.ingress().clone(), server_a.base.to_string()),
        (b.ingress().clone(), server_b.base.to_string()),
    ];
    let first = Forwarder::new(
        routes.clone(),
        keys()?,
        Default::default(),
        "test-only-token-012345678901234567890".into(),
    )?;
    let second = Forwarder::new(
        routes.into_iter().rev().collect(),
        keys()?,
        Default::default(),
        "test-only-token-012345678901234567890".into(),
    )?;
    let ingress_a = Server::start(first.router()).await?;
    let ingress_b = Server::start(second.router()).await?;
    let (entered, resume) = faults.pause_data_write();
    let (pause_apply, pause_requested) = tokio::sync::oneshot::channel();
    let (apply_paused, paused) = tokio::sync::oneshot::channel();
    let task_a = tokio::spawn(async move {
        let work = ra.run(std::future::pending());
        tokio::pin!(work);
        tokio::select! {
            result = &mut work => result,
            _ = pause_requested => {
                let _ = apply_paused.send(());
                // Retain the runtime without polling its applier/publisher.
                // Aborting this task below still drops the real runtime.
                std::future::pending::<Result<()>>().await
            }
        }
    });
    let (stop_b, task_b) = run_owner(rb);
    let writer_a = shards[0]
        .client(
            &root.path().join("producer-a"),
            ingress_a.base.clone(),
            &layout,
            &spec,
            1,
        )
        .await?;
    let other_a = shards[0]
        .client(
            &root.path().join("producer-a-other"),
            ingress_b.base.clone(),
            &layout,
            &spec,
            2,
        )
        .await?;
    let writer_b = shards[1]
        .client(
            &root.path().join("producer-b"),
            ingress_b.base.clone(),
            &layout,
            &spec,
            1,
        )
        .await?;
    let ack_a = writer_a
        .append(0, shards[0].wire(&spec, &input[0])?)
        .await?;
    assert_eq!(ack_a.completion.result, MutationResult::Applied);
    tokio::time::timeout(DEADLINE, entered).await??;
    assert_eq!(other_a.selected_full_body().await?.rows, Some(0));
    check(&a, &spec, &input[0]).await?;
    http_check(&ingress_b.base, &spec, &input[0]).await?;

    // A confirmed miss and unavailable read admission are distinct HTTP results.
    let missing = crate::http::annotations::PointRequest {
        turn: spec.business_key(&[&sessions[0], "absent"], 4096)?,
    };
    let endpoint = ingress_b.base.join(&format!(
        "/api/v1/annotations/{}/read-body",
        binding().qualified_table
    ))?;
    assert_eq!(
        transport()?
            .post(endpoint.clone())
            .json(&missing)
            .send()
            .await?
            .status(),
        reqwest::StatusCode::NO_CONTENT
    );
    let held = read_safety::idle_budget(&uploads).await?;
    assert_eq!(
        transport()?
            .post(endpoint.clone())
            .json(&missing)
            .send()
            .await?
            .status(),
        reqwest::StatusCode::SERVICE_UNAVAILABLE
    );
    drop(held);
    http_check(&ingress_b.base, &spec, &input[0]).await?;
    read_safety::admission_and_response_lifetime(
        &oa,
        &a,
        &uploads,
        &missing,
        &crate::http::annotations::PointRequest {
            turn: spec.row_key(&input[0], 0, 4096)?,
        },
    )
    .await?;

    // Independent producer through the other ingress cannot replace a body
    // already visible in frozen staging, even with different payload bytes.
    let changed = body(&sessions[0], "same-id", 99)?;
    let duplicate = other_a.append(0, shards[0].wire(&spec, &changed)?).await?;
    assert_eq!(duplicate.completion.result, MutationResult::Applied);
    check(&a, &spec, &input[0]).await?;
    let wrong = request(
        &a,
        &producer(&a, 3).await?,
        1,
        shards[0].wire(&spec, &input[1])?,
    )
    .await;
    let error = wrong.err().context("foreign session entered this shard")?;
    assert!(error.to_string().contains("another shard"), "{error:#}");
    let ack_b = writer_b
        .append(0, shards[1].wire(&spec, &input[1])?)
        .await?;
    shards[1].drained(&c, &b, ack_b.completion.index).await?;
    assert_eq!(writer_b.selected_full_body().await?.rows, Some(1));
    assert_eq!(other_a.selected_full_body().await?.rows, Some(0));
    check(&b, &spec, &input[1]).await?;
    http_check(&ingress_a.base, &spec, &input[1]).await?;

    pause_apply.send(()).unwrap();
    tokio::time::timeout(DEADLINE, paused).await??;
    // Commit a later request without applying it. Reads of the acknowledged
    // body wait only for this owner's acquisition cut, not the current WAL tail.
    let pending = request(
        &a,
        &producer(&a, 4).await?,
        1,
        shards[0].wire(&spec, &input[0])?,
    )
    .await?;
    let accepted = a.ingress().append(&pending).await?;
    assert!(accepted.position.index > a.observation().await.0.applied);
    tokio::time::timeout(
        Duration::from_secs(2),
        http_check(&ingress_b.base, &spec, &input[0]),
    )
    .await??;
    drop(pending);
    // Kill A before its frozen upload completes. B must still accept and
    // materialize another body without A advancing its publication cut.
    task_a.abort();
    assert!(task_a.await.unwrap_err().is_cancelled());
    drop(resume);
    let extra_b = body(&sessions[1], "new-id", 11)?;
    let extra = writer_b.append(0, shards[1].wire(&spec, &extra_b)?).await?;
    shards[1].drained(&c, &b, extra.completion.index).await?;
    assert_eq!(writer_b.selected_full_body().await?.rows, Some(2));
    check(&b, &spec, &extra_b).await?;
    writer_a.shutdown().await?;
    other_a.shutdown().await?;
    writer_b.shutdown().await?;
    first.shutdown().await;
    second.shutdown().await;
    ingress_a.shutdown().await?;
    ingress_b.shutdown().await?;
    oa.shutdown().await;
    ob.shutdown().await;
    server_a.shutdown().await?;
    server_b.shutdown().await?;
    stop_b.send(()).unwrap();
    task_b.await??;
    drop((a, b));
    c.shutdown().await?;

    // Reopen all three WAL directories and use fresh Lance storage sessions.
    // A replays acknowledged but unpublished bodies; B resumes its own cut.
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    for physical in &mut shards {
        physical.storage = fault_store::storage(
            &root.path().join(&physical.binding.destination.dataset),
            Arc::default(),
        )
        .await?;
    }
    let uploads = MergeMemoryBudget::new(8 << 20);
    let (a, ra) = shards[0].owner(&c, &layout, &spec, uploads.clone()).await?;
    let (b, rb) = shards[1].owner(&c, &layout, &spec, uploads).await?;
    let (stop_a, task_a) = run_owner(ra);
    let (stop_b, task_b) = run_owner(rb);
    check(&a, &spec, &input[0]).await?;
    check(&b, &spec, &input[1]).await?;
    check(&b, &spec, &extra_b).await?;
    shards[0]
        .drained(&c, &a, duplicate.completion.index)
        .await?;
    shards[1].drained(&c, &b, extra.completion.index).await?;
    let selected_a = a.selected_full_body(DEADLINE).await?;
    let selected_b = b.selected_full_body(DEADLINE).await?;
    assert_eq!(selected_a.rows, Some(1));
    assert_eq!(selected_b.rows, Some(2));
    assert_ne!(selected_a.source, selected_b.source);
    assert_eq!(selected_a.source.shard, 0);
    assert_eq!(selected_b.source.shard, 1);
    let original_lane = producer(&a, 1).await?;
    let repeated = request(&a, &original_lane, 1, shards[0].wire(&spec, &input[0])?).await?;
    assert_eq!(a.submit(&repeated, DEADLINE).await?, ack_a.completion);
    check(&a, &spec, &input[0]).await?;
    stop_a.send(()).unwrap();
    stop_b.send(()).unwrap();
    task_a.await??;
    task_b.await??;
    c.shutdown().await
}
