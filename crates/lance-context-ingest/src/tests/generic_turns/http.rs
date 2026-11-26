use super::*;
use crate::{
    annotations::runtime::{Configuration, Runtime},
    http::{
        annotations::{Forwarder, MutationReply, Owner, PointRequest},
        journal::{Config, DurableMutationClient},
    },
    materializer::{Acquisition, Options},
};
use axum::{
    body::to_bytes,
    extract::{Request, State},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Router,
};
use lance_context_wal::client_journal::Settings;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};

mod full_body;

#[derive(Default)]
struct LostReply {
    once: AtomicBool,
    completion: Mutex<Option<MutationReply>>,
    replacement_once: AtomicBool,
}

async fn lose_reply(State(lost): State<Arc<LostReply>>, request: Request, next: Next) -> Response {
    let update = request.uri().path().ends_with("/update");
    let replace = request.uri().path().ends_with("/replace-producer");
    let response = next.run(request).await;
    if replace && response.status() == axum::http::StatusCode::OK {
        let body = to_bytes(response.into_body(), 32 << 10).await.unwrap();
        let reply: crate::http::annotations::ReplaceProducerReply =
            serde_json::from_slice(&body).unwrap();
        if matches!(
            reply,
            crate::http::annotations::ReplaceProducerReply::Opened { .. }
        ) && !lost.replacement_once.swap(true, Ordering::AcqRel)
        {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "lost replacement reply",
            )
                .into_response();
        }
        return (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response();
    }
    if update
        && response.status() == axum::http::StatusCode::OK
        && !lost.once.swap(true, Ordering::AcqRel)
    {
        let body = to_bytes(response.into_body(), 32 << 10).await.unwrap();
        *lost.completion.lock().unwrap() = Some(serde_json::from_slice(&body).unwrap());
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(
                serde_json::json!({"error":"test_lost_reply","operation_outcome":"unknown"}),
            ),
        )
            .into_response();
    }
    response
}

struct Server {
    base: reqwest::Url,
    stop: oneshot::Sender<()>,
    task: JoinHandle<std::io::Result<()>>,
}
impl Server {
    async fn start(router: Router) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base = format!("http://{}/", listener.local_addr()?).parse()?;
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });
        Ok(Self { base, stop, task })
    }
    async fn shutdown(self) -> Result<()> {
        let _ = self.stop.send(());
        self.task.await??;
        Ok(())
    }
}

fn keys() -> Result<ReceiptKeys> {
    ReceiptKeys::new(1, [(1, [0x42; 32])])
}

async fn client(
    root: &std::path::Path,
    base: reqwest::Url,
    spec: &TurnIngestionSpec,
) -> Result<DurableMutationClient> {
    client_with_open(
        root,
        base,
        spec,
        ProducerOpen {
            bucket: 0,
            slot: 1,
            expected_generation: 0,
            operation: Uuid::from_u128(101),
        },
    )
    .await
}

async fn client_with_open(
    root: &std::path::Path,
    base: reqwest::Url,
    spec: &TurnIngestionSpec,
    open: ProducerOpen,
) -> Result<DurableMutationClient> {
    DurableMutationClient::open(
        Config {
            directory: root.join("client-journal"),
            opens: vec![open],
            base,
            request_timeout: DEADLINE,
            journal: Settings {
                state_bytes: 16 << 20,
                queue_bytes: 8 << 20,
                batch_bytes: 2 << 20,
                file_bytes: 64 << 10,
                log_bytes: 96 << 20,
                ..Default::default()
            },
        },
        &annotation_publication::one_shard(),
        binding(),
        spec,
        &fixture::limits(),
        Shard { bucket: 0 },
        reqwest::Client::builder()
            .retry(reqwest::retry::never())
            .redirect(reqwest::redirect::Policy::none())
            .build()?,
        MergeMemoryBudget::new(64 << 20),
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn durable_client_recovers_lost_and_rejected_replies_across_ingresses() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(90), exercise()).await?
}

async fn exercise() -> Result<()> {
    let root = tempfile::tempdir()?;
    let source = bodies()?.remove(0);
    let spec = TurnIngestionSpec::new(
        source.schema(),
        vec!["session_id".into(), "id".into()],
        "session_id",
    )?;
    let ids = spec.prepare(source.clone(), 4096)?.keys().to_vec();
    let initial = fixture::patch(
        &ids.iter().map(String::as_str).collect::<Vec<_>>(),
        Some("pending"),
        None,
    )?;
    let create = encode(mixed::wire::creates_for(
        &spec,
        &source,
        &initial,
        &fixture::limits(),
    )?)?;
    let patch = |key: &str| {
        encode(mixed::wire::patches_for(
            &spec,
            &fixture::patch(&[key], Some("reviewed"), Some(0.9))?,
            &fixture::limits(),
        )?)
    };
    let c = network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
    let narrow = fixture::storage(root.path(), "narrow.lance").await?;
    let body = fixture::storage(root.path(), "body.lance").await?;
    provision_turn_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        narrow.clone(),
        body.clone(),
        Identity {
            owner: Uuid::new_v4(),
            acquisition: Uuid::new_v4(),
            manifest: Uuid::new_v4(),
        },
        fixture::limits(),
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let state = annotation_publication::state(&c).await?;
    let (service, runtime) = Runtime::connect_turn_shard(
        c.client.clone(),
        annotation_publication::one_shard(),
        binding(),
        narrow,
        body,
        Configuration {
            limits: fixture::limits(),
            upload_budget: MergeMemoryBudget::new(8 << 20),
            apply_budget: MergeMemoryBudget::new(8 << 20),
            publication_budget: MergeMemoryBudget::new(8 << 20),
            publication: Options {
                max_source_bytes: 1 << 20,
                min_source_bytes: 1 << 20,
                max_records: 2,
                max_batch_wait: Duration::from_millis(20),
                ..Default::default()
            },
        },
        Acquisition {
            owner: Uuid::new_v4(),
            request: Uuid::new_v4(),
            expected_epoch: state.authority.epoch,
        },
        Shard { bucket: 0 },
        &spec,
    )
    .await?;
    let owner = Owner::new(&runtime, keys()?, Default::default())?;
    let (stop, owner_task) = annotation_runtime::run_owner(runtime);
    let lost = Arc::new(LostReply::default());
    let server = Server::start(
        owner
            .router()
            .layer(middleware::from_fn_with_state(lost.clone(), lose_reply)),
    )
    .await?;
    let forward = || {
        Forwarder::new(
            vec![(service.ingress().clone(), server.base.to_string())],
            keys()?,
            Default::default(),
            "test-only-token-012345678901234567890".into(),
        )
    };
    let first = forward()?;
    let second = forward()?;
    let a = Server::start(first.router()).await?;
    let b = Server::start(second.router()).await?;
    let writer = client(root.path(), a.base.clone(), &spec).await?;
    assert_eq!(writer.stage(0, create.clone()).await?, 1);
    writer.shutdown().await?;
    assert!(service.get(&ids[0], DEADLINE).await?.is_none());
    let writer = client(root.path(), a.base.clone(), &spec).await?;
    assert!(
        writer.resume(0).await.is_err(),
        "response loss must leave the original intent pending"
    );
    assert!(writer.last_completed(0).await?.is_none());
    assert!(writer.stage(0, patch(&ids[0])?).await.is_err());
    let original = lost
        .completion
        .lock()
        .unwrap()
        .clone()
        .context("missing lost completion")?;
    writer.shutdown().await?;
    let writer = client(root.path(), b.base.clone(), &spec).await?;
    assert_eq!(writer.resume(0).await?, Some(original.clone()));
    assert!(writer.resume(0).await?.is_none());
    let missing = spec.business_key(&["missing-session", "missing-turn"], 4096)?;
    let rejected = writer.append(0, patch(&missing)?).await?;
    assert!(matches!(
        rejected.completion.result,
        MutationResult::Rejected(_)
    ));
    assert_eq!(rejected.completion.meta.sequence, 2);
    writer.shutdown().await?;
    let writer = client(root.path(), a.base.clone(), &spec).await?;
    assert_eq!(writer.last_completed(0).await?, Some(rejected));
    assert!(writer.resume(0).await?.is_none());
    assert_eq!(
        writer
            .append(0, patch(&ids[0])?)
            .await?
            .completion
            .meta
            .sequence,
        3
    );
    let duplicate = writer.append(0, create.clone()).await?;
    assert_eq!(duplicate.completion.meta.sequence, 4);
    let read = reqwest::Client::new()
        .post(b.base.join("api/v1/annotations/rollouts/read")?)
        .json(&PointRequest {
            turn: ids[0].clone(),
        })
        .send()
        .await?
        .error_for_status()?
        .json::<Option<lance_context_api::annotation::TurnAnnotation>>()
        .await?
        .unwrap();
    assert_eq!(read.status.as_deref(), Some("reviewed"));
    assert_eq!(read.score, Some(0.9));
    annotation_runtime::drained(&c, &service, duplicate.completion.index).await?;
    let selected = writer.selected_body().await?;
    assert!(selected.through >= duplicate.completion.index);
    let materialized = lance::dataset::builder::DatasetBuilder::from_uri(&selected.body.uri)
        .with_version(selected.body.version)
        .load()
        .await?;
    assert_eq!(materialized.count_rows(None).await?, 2);
    assert_eq!(
        service.get_turn(&ids[0], DEADLINE).await?.unwrap().body(),
        &source.slice(0, 1)
    );
    assert_eq!(
        reqwest::Client::new()
            .post(b.base.join("api/v1/annotations/rollouts/selected-body")?)
            .json(&serde_json::json!({"bucket":1}))
            .send()
            .await?
            .status(),
        reqwest::StatusCode::BAD_REQUEST
    );
    // Abandon an executor journal after a durable tail, with a scheduler-owned
    // control intent. Lose the successful replacement reply, then retry via
    // the other ingress without reacquiring the now-stale predecessor.
    use crate::http::{
        annotations::ReplaceProducerReply,
        journal::{ProducerControlClient, ReplacementIntent},
    };
    lost.once.store(false, Ordering::Release);
    assert!(writer.append(0, create.clone()).await.is_err());
    assert_eq!(
        writer
            .last_completed(0)
            .await?
            .unwrap()
            .completion
            .meta
            .sequence,
        4
    );
    let abandoned_index = lost
        .completion
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .completion
        .index;
    annotation_runtime::drained(&c, &service, abandoned_index).await?;
    writer.shutdown().await?;
    let control = |base| {
        ProducerControlClient::new(
            base,
            reqwest::Client::builder()
                .retry(reqwest::retry::never())
                .build()
                .unwrap(),
            DEADLINE,
            &annotation_publication::one_shard(),
            &binding(),
            &spec,
            &fixture::limits(),
            Shard { bucket: 0 },
        )
    };
    let old_open = ProducerOpen {
        bucket: 0,
        slot: 1,
        expected_generation: 0,
        operation: Uuid::from_u128(101),
    };
    let ctl = control(a.base.clone())?;
    let intent = ctl.prepare_replacement(&old_open, Uuid::new_v4()).await?;
    let persisted = serde_json::to_vec(&intent)?;
    assert!(
        ctl.execute(&intent).await.is_err(),
        "injected successful reply loss"
    );
    assert!(lost.replacement_once.load(Ordering::Acquire));
    let intent: ReplacementIntent = serde_json::from_slice(&persisted)?;
    let ctl = control(b.base.clone())?;
    let ReplaceProducerReply::Opened { open, .. } = ctl.execute(&intent).await? else {
        panic!("retired predecessor")
    };
    assert_eq!(open.expected_generation, 1);
    let successor_root = root.path().join("successor");
    std::fs::create_dir(&successor_root)?;
    let successor = client_with_open(&successor_root, b.base.clone(), &spec, open.clone()).await?;
    let replayed = successor.append(0, create).await?;
    assert_eq!(replayed.completion.meta.sequence, 1);
    assert_eq!(replayed.completion.result, MutationResult::Applied);
    annotation_runtime::drained(&c, &service, replayed.completion.index).await?;
    assert_eq!(selected.body, successor.selected_body().await?.body);
    assert_eq!(
        service.get(&ids[0], DEADLINE).await?.unwrap().score,
        Some(0.9)
    );
    let ReplaceProducerReply::Opened { open: again, .. } = ctl.execute(&intent).await? else {
        panic!("same intent")
    };
    assert_eq!(
        open, again,
        "retry cannot allocate another generation after successor writes"
    );
    // A competing intent for the stale predecessor cannot revoke the successor.
    let mut conflicting = serde_json::to_value(&intent)?;
    let other = serde_json::to_value(Uuid::new_v4())?;
    conflicting["successor"]["operation"] = other.clone();
    conflicting["request"]["operation"] = other;
    let conflicting: ReplacementIntent = serde_json::from_value(conflicting)?;
    assert!(ctl
        .execute(&conflicting)
        .await
        .unwrap_err()
        .is::<ProducerReplacementFenced>());
    successor.shutdown().await?;
    first.shutdown().await;
    second.shutdown().await;
    a.shutdown().await?;
    b.shutdown().await?;
    owner.shutdown().await;
    server.shutdown().await?;
    let _ = stop.send(());
    owner_task.await??;
    c.shutdown().await
}
