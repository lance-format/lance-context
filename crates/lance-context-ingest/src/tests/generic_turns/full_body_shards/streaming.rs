use super::*;
use crate::{
    http::journal::AppendResponseError,
    streaming::{Config, Lane, ShardLanes, StreamingProducer},
};
use axum::{
    extract::Request,
    http::StatusCode,
    middleware::{self, Next},
    response::IntoResponse,
    Json,
};
use lance_context_core::MergeReservation;
use lance_context_wal::replication::runtime::client::payload_memory::{Pool, Reservation};
use std::collections::BTreeMap;

struct Memory(Arc<MergeMemoryBudget>);

struct Charge(MergeReservation);

impl Pool for Memory {
    fn try_reserve(&self, bytes: usize) -> Result<Option<Box<dyn Reservation>>> {
        Ok(self
            .0
            .try_reserve(bytes)
            .map(|charge| Box::new(Charge(charge)) as _))
    }
}

impl Reservation for Charge {
    fn bytes(&self) -> usize {
        self.0.bytes()
    }

    fn set_backed_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn terminal_applied_error_unblocks_fence_with_another_lane_pending() -> Result<()> {
    tokio::time::timeout(DEADLINE, exercise_error()).await?
}

async fn exercise_error() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut definition = (*layout()).clone();
    definition.buckets = vec![10, 20];
    let layout = Arc::new(definition);
    let input = (0..2)
        .map(|bucket| {
            let session = (0..1000)
                .map(|i| format!("session-{i}"))
                .find(|s| Shard::for_session(&layout, &binding(), s).unwrap().bucket == bucket)
                .context("no session for shard")?;
            body(&session, "fence-receipt", 7)
        })
        .collect::<Result<Vec<_>>>()?;
    let spec = TurnIngestionSpec::new(
        input[0].schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let c = network::Cluster::open_signed(root.path(), layout.clone()).await?;
    let pending_entered = Arc::new(tokio::sync::Notify::new());
    let mut owners = Vec::new();
    let mut lanes = BTreeMap::new();
    for bucket in 0..2 {
        let mut table = binding();
        table.table = Uuid::new_v4();
        table.incarnation = Uuid::new_v4();
        table.destination.dataset = format!("fence-body-{bucket}.lance");
        let shard = Shard { bucket };
        let storage = fault_store::storage(
            &root.path().join(&table.destination.dataset),
            Arc::default(),
        )
        .await?;
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
        let physical = Physical {
            binding: table,
            shard,
            storage,
        };
        let (_, runtime) = physical
            .owner(&c, &layout, &spec, MergeMemoryBudget::new(32 << 20))
            .await?;
        let pending_entered = pending_entered.clone();
        let router = Owner::new(&runtime, keys()?, Default::default())?
            .router()
            .layer(middleware::from_fn(move |request: Request, next: Next| {
                let pending_entered = pending_entered.clone();
                async move {
                    if !request.uri().path().ends_with("/applied") {
                        return next.run(request).await;
                    }
                    let (status, error, outcome) = if bucket == 0 {
                        pending_entered.notify_one();
                        (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "applied_pending",
                            "unknown",
                        )
                    } else {
                        pending_entered.notified().await;
                        (StatusCode::CONFLICT, "prior_request_rejected", "rejected")
                    };
                    (
                        status,
                        Json(serde_json::json!({"error": error, "operation_outcome": outcome})),
                    )
                        .into_response()
                }
            }));
        let server = Server::start(router).await?;
        let client = physical
            .client(
                &root.path().join(format!("fence-producer-{bucket}")),
                server.base.clone(),
                &layout,
                &spec,
                1,
            )
            .await?;
        lanes.insert(
            bucket,
            ShardLanes {
                binding: physical.binding,
                lanes: vec![Lane {
                    client: Arc::new(client),
                    key: 0,
                }],
            },
        );
        owners.push((runtime, server));
    }
    let producer = StreamingProducer::open(Config {
        spec,
        limits: fixture::limits(),
        layout,
        shards: lanes,
        memory: Arc::new(Memory(MergeMemoryBudget::new(64 << 20))),
        buffer_ram_bytes: 8 << 20,
        spill_dir: root.path().join("stream-spill"),
        spill_bytes: 0,
        applied_parallelism: 2,
        transient: Arc::new(|err| {
            err.downcast_ref::<AppendResponseError>()
                .is_some_and(|err| err.code == "applied_pending")
        }),
    })?;
    for batch in input {
        producer.send(batch).await?;
    }
    while producer.metrics().records < 2 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let error = tokio::time::timeout(Duration::from_secs(2), producer.fence())
        .await
        .context("terminal error was hidden by a pending lane")?
        .expect_err("terminal receipt error must fail the fence");
    assert_eq!(
        error
            .downcast_ref::<AppendResponseError>()
            .map(|e| e.code.as_str()),
        Some("prior_request_rejected")
    );
    assert!(
        producer.fence().await.is_err(),
        "a later fence must see the failure"
    );
    producer.close().await?;
    for (runtime, server) in owners {
        server.shutdown().await?;
        drop(runtime);
    }
    c.shutdown().await
}
