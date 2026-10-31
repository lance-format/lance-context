//! Independent physical tables share a logical name and the existing RF3 WAL.
//! This tests protocol isolation/recovery, not multi-host throughput or ingress routing.
use super::*;
use super::{annotations::request, mixed_annotations as fixture};
use crate::{
    annotations::{
        mixed,
        runtime::{Configuration, Runtime},
        MutationResult, Service, Shard,
    },
    bootstrap::{annotations::mixed::provision_shard, Identity},
    materializer::{Acquisition, Options, Storage},
};

const DEADLINE: Duration = Duration::from_secs(30);

mod compaction;

pub(super) struct Physical {
    pub(super) shard: Shard,
    pub(super) binding: TableBinding,
    pub(super) narrow: Storage,
    pub(super) body: Storage,
}

impl Physical {
    pub(super) async fn provision(
        c: &network::Cluster,
        layout: &Arc<Layout>,
        root: &std::path::Path,
        bucket: u32,
    ) -> Result<Self> {
        Self::provision_with_binding(c, layout, root, bucket, binding()).await
    }

    async fn provision_with_binding(
        c: &network::Cluster,
        layout: &Arc<Layout>,
        root: &std::path::Path,
        bucket: u32,
        mut binding: TableBinding,
    ) -> Result<Self> {
        let shard = Shard { bucket };
        binding.table = Uuid::new_v4();
        binding.incarnation = Uuid::new_v4();
        binding.destination.dataset = format!("shard-{bucket}.lance");
        let narrow = fixture::storage(root, &binding.destination.dataset).await?;
        let body = fixture::storage(root, &format!("body-{bucket}.lance")).await?;
        provision_shard(
            c.client.clone(),
            layout.clone(),
            binding.clone(),
            narrow.clone(),
            body.clone(),
            Identity {
                owner: Uuid::new_v4(),
                acquisition: Uuid::new_v4(),
                manifest: Uuid::new_v4(),
            },
            fixture::limits(),
            shard,
        )
        .await?;
        Ok(Self {
            shard,
            binding,
            narrow,
            body,
        })
    }

    pub(super) async fn view(&self, c: &network::Cluster) -> Result<Box<p::Table>> {
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
            value: query::Value::Publisher(Some(view)),
            ..
        }) = response.reply
        else {
            bail!("missing shard publisher state");
        };
        Ok(view)
    }

    pub(super) async fn owner(
        &self,
        c: &network::Cluster,
        layout: &Arc<Layout>,
        wait: Duration,
    ) -> Result<(Service, Runtime)> {
        self.owner_as(c, layout, wait, self.shard).await
    }

    async fn owner_as(
        &self,
        c: &network::Cluster,
        layout: &Arc<Layout>,
        wait: Duration,
        shard: Shard,
    ) -> Result<(Service, Runtime)> {
        Runtime::connect_mixed_shard(
            c.client.clone(),
            layout.clone(),
            self.binding.clone(),
            self.narrow.clone(),
            self.body.clone(),
            Configuration {
                limits: fixture::limits(),
                upload_budget: MergeMemoryBudget::new(8 << 20),
                apply_budget: MergeMemoryBudget::new(8 << 20),
                publication_budget: MergeMemoryBudget::new(8 << 20),
                publication: Options {
                    max_source_bytes: 1 << 20,
                    min_source_bytes: 1 << 20,
                    max_records: 64,
                    max_batch_wait: wait,
                    ..Default::default()
                },
            },
            Acquisition {
                owner: Uuid::new_v4(),
                request: Uuid::new_v4(),
                expected_epoch: self.view(c).await?.authority.epoch,
            },
            shard,
        )
        .await
    }

    pub(super) fn turn(&self, layout: &Layout) -> String {
        (0..1000)
            .map(|i| format!("turn-{i}"))
            .find(|id| Shard::for_turn(layout, &self.binding, id).unwrap() == self.shard)
            .unwrap()
    }

    pub(super) fn encode(&self, batch: RecordBatch) -> Result<Bytes> {
        let bound = self.shard.bind_batch(&batch)?;
        for (old, new) in batch.columns().iter().zip(bound.columns()) {
            assert!(
                Arc::ptr_eq(old, new),
                "binding must share every Arrow array"
            );
        }
        fixture::encoded(bound)
    }

    pub(super) fn patch(
        &self,
        id: &str,
        status: Option<&str>,
        score: Option<f32>,
    ) -> Result<Bytes> {
        self.encode(mixed::wire::patches(
            &fixture::patch(&[id], status, score)?,
            &fixture::limits(),
        )?)
    }

    pub(super) async fn drained(
        &self,
        c: &network::Cluster,
        layout: &Layout,
        service: &Service,
        through: u64,
    ) -> Result<()> {
        let group = layout.group_for_bucket(self.shard.bucket)?;
        tokio::time::timeout(DEADLINE, async {
            loop {
                let (o, pressured) = service.observation().await;
                let view = self.view(c).await?;
                if o.materialized >= through
                    && o.retained_bytes == 0
                    && o.retained_metadata_bytes == 0
                    && !pressured
                    && service.source_observation().records == 0
                    && view.progress.pending.is_none()
                    && view.progress.checkpoint.get(&group).copied().unwrap_or(0) >= through
                {
                    assert_eq!(
                        view.progress.checkpoint.len(),
                        1,
                        "shard must advance only its source group"
                    );
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await??;
        Ok(())
    }
}

async fn producer(service: &Service) -> Result<Producer> {
    service
        .open_producer(ProducerOpen {
            bucket: service.bucket(),
            slot: 1,
            expected_generation: 0,
            operation: Uuid::new_v4(),
        })
        .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn annotation_shards_reject_foreign_batches_and_drain_independently_across_recovery(
) -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut definition = (*layout()).clone();
    definition.buckets = vec![10, 20];
    let layout = Arc::new(definition);
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let a = Physical::provision(&c, &layout, root.path(), 0).await?;
    let b = Physical::provision(&c, &layout, root.path(), 1).await?;
    assert_eq!(a.binding.qualified_table, b.binding.qualified_table);
    let ta = a.turn(&layout);
    let tb = b.turn(&layout);
    assert_ne!(ta, tb);
    // Both owners hold their ACKed WAL prefix until explicitly resumed below.
    let (sa, ra) = a.owner(&c, &layout, Duration::from_secs(60)).await?;
    let (sb, rb) = b.owner(&c, &layout, Duration::from_secs(60)).await?;
    let (_stop_a, task_a) = annotation_runtime::run_owner(ra);
    let (stop_b, task_b) = annotation_runtime::run_owner(rb);
    let pa = producer(&sa).await?;
    let pb = producer(&sb).await?;
    let create_a = request(
        &sa,
        &pa,
        1,
        a.encode(fixture::create(&[&ta], &["same-session"], &[7])?)?,
    )
    .await?;
    let create_b = request(
        &sb,
        &pb,
        1,
        b.encode(fixture::create(&[&tb], &["same-session"], &[9])?)?,
    )
    .await?;
    let (ack_a, ack_b) = tokio::try_join!(
        sa.submit(&create_a, DEADLINE),
        sb.submit(&create_b, DEADLINE)
    )?;
    assert_eq!(ack_a.meta.bucket, 0);
    assert_eq!(ack_b.meta.bucket, 1);
    assert_eq!(ack_a.result, MutationResult::Applied);
    assert_eq!(ack_b.result, MutationResult::Applied);

    // An atomic batch with a valid first row and foreign second row must never
    // consume a producer sequence or partially modify the valid row.
    let wrong = a.encode(mixed::wire::patches(
        &fixture::patch(&[&ta, &tb], Some("wrong"), None)?,
        &fixture::limits(),
    )?)?;
    let error = request(&sa, &pa, 2, wrong)
        .await
        .err()
        .context("foreign batch accepted")?;
    assert!(error.to_string().contains("another shard"));
    fixture::check(&sa, &ta, "same-session", 7, "pending", None).await?;
    assert!(sa.get(&tb, DEADLINE).await.is_err());
    assert!(sb.get_turn(&ta, DEADLINE).await.is_err());
    let unbound = fixture::encoded(fixture::create(&[&ta], &["same-session"], &[7])?)?;
    assert!(request(&sa, &pa, 2, unbound).await.is_err());
    let patch_a = request(&sa, &pa, 2, a.patch(&ta, Some("reviewed"), None)?).await?;
    let patch_ack_a = sa.submit(&patch_a, DEADLINE).await?;
    let patch_b = request(&sb, &pb, 2, b.patch(&tb, Some("reviewed"), None)?).await?;
    let patch_ack_b = sb.submit(&patch_b, DEADLINE).await?;
    fixture::check(&sa, &ta, "same-session", 7, "reviewed", None).await?;
    fixture::check(&sb, &tb, "same-session", 9, "reviewed", None).await?;
    let before_a = a.view(&c).await?;
    assert!(before_a.progress.checkpoint.is_empty());
    assert!(sa.source_observation().records > 0);

    // Stop one owner with unmaterialized ACKs. The other resumes, accepts a
    // further field update and publishes without waiting for the stopped shard.
    task_a.abort();
    assert!(task_a.await.unwrap_err().is_cancelled());
    drop(sa);
    stop_b.send(()).unwrap();
    task_b.await??;
    drop(sb);
    let (sb, rb) = b.owner(&c, &layout, Duration::from_millis(20)).await?;
    let (stop_b, task_b) = annotation_runtime::run_owner(rb);
    assert_eq!(sb.submit(&patch_b, DEADLINE).await?, patch_ack_b);
    let score_b = request(&sb, &pb, 3, b.patch(&tb, None, Some(0.9))?).await?;
    let score_ack_b = sb.submit(&score_b, DEADLINE).await?;
    b.drained(&c, &layout, &sb, score_ack_b.index).await?;
    fixture::check(&sb, &tb, "same-session", 9, "reviewed", Some(0.9)).await?;
    assert_eq!(
        a.view(&c).await?,
        before_a,
        "other shard publication must not advance or fence A"
    );
    let before_b = b.view(&c).await?;

    // The durable schema contract forbids rebinding an existing physical table
    // to another bucket, including before acquiring a new owner epoch.
    assert!(a
        .owner_as(&c, &layout, Duration::from_millis(20), b.shard)
        .await
        .is_err());
    assert_eq!(a.view(&c).await?, before_a);
    let (sa, ra) = a.owner(&c, &layout, Duration::from_millis(20)).await?;
    let (stop_a, task_a) = annotation_runtime::run_owner(ra);
    assert_eq!(sa.submit(&patch_a, DEADLINE).await?, patch_ack_a);
    fixture::check(&sa, &ta, "same-session", 7, "reviewed", None).await?;
    a.drained(&c, &layout, &sa, patch_ack_a.index).await?;
    assert_eq!(b.view(&c).await?, before_b);
    let final_a = a.view(&c).await?;
    let final_b = b.view(&c).await?;
    assert_ne!(final_a.authority.table, final_b.authority.table);
    stop_a.send(()).unwrap();
    stop_b.send(()).unwrap();
    task_a.await??;
    task_b.await??;
    drop((sa, sb));
    c.shutdown().await?;

    // Reopen all three WAL directories and use fresh Lance sessions. Recovery
    // must use each physical table's own cut and retained request results.
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let a = Physical {
        narrow: fixture::storage(root.path(), "shard-0.lance").await?,
        body: fixture::storage(root.path(), "body-0.lance").await?,
        ..a
    };
    let b = Physical {
        narrow: fixture::storage(root.path(), "shard-1.lance").await?,
        body: fixture::storage(root.path(), "body-1.lance").await?,
        ..b
    };
    assert_eq!(
        a.view(&c).await?.progress.checkpoint,
        final_a.progress.checkpoint
    );
    assert_eq!(
        b.view(&c).await?.progress.checkpoint,
        final_b.progress.checkpoint
    );
    let (sa, ra) = a.owner(&c, &layout, Duration::from_millis(20)).await?;
    let (sb, rb) = b.owner(&c, &layout, Duration::from_millis(20)).await?;
    let (stop_a, task_a) = annotation_runtime::run_owner(ra);
    let (stop_b, task_b) = annotation_runtime::run_owner(rb);
    fixture::check(&sa, &ta, "same-session", 7, "reviewed", None).await?;
    fixture::check(&sb, &tb, "same-session", 9, "reviewed", Some(0.9)).await?;
    assert_eq!(sa.submit(&patch_a, DEADLINE).await?, patch_ack_a);
    assert_eq!(sb.submit(&score_b, DEADLINE).await?, score_ack_b);
    stop_a.send(()).unwrap();
    stop_b.send(()).unwrap();
    task_a.await??;
    task_b.await??;
    c.shutdown().await?;
    Ok(())
}
