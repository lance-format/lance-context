use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use axum::{extract::State, http::StatusCode, routing::get, routing::post, Json, Router};
use lance_context_api::{
    ColumnSpec, ColumnType, ContextStoreApi, CreateContextRequest, CreateDatagenStoreRequest,
    CreateGenericStoreRequest, CreateRolloutStoreRequest, DatagenStoreApi, GenericStoreApi,
    RolloutStoreApi, SchemaSpec,
};
use lance_context_client::{
    ClientError, RemoteContextStore, RemoteDatagenStore, RemoteGenericStore, RemoteRolloutStore,
};
use serde_json::{json, Value};
use tokio::{net::TcpListener, sync::Barrier, task::JoinHandle, time::timeout};

#[derive(Clone, Copy, Debug)]
enum StoreKind {
    Context,
    Rollout,
    Generic,
    Datagen,
}

const STORE_KINDS: [StoreKind; 4] = [
    StoreKind::Context,
    StoreKind::Rollout,
    StoreKind::Generic,
    StoreKind::Datagen,
];

fn schema(extra_column: &str) -> SchemaSpec {
    SchemaSpec::new(vec![
        (
            "id".into(),
            ColumnSpec::required(ColumnType::String { large: false }),
        ),
        (extra_column.into(), ColumnSpec::new(ColumnType::Int64)),
    ])
}

impl StoreKind {
    fn path(self) -> &'static str {
        match self {
            Self::Context => "/api/v1/contexts",
            Self::Rollout => "/api/v1/rollouts",
            Self::Generic => "/api/v1/generic",
            Self::Datagen => "/api/v1/datagen",
        }
    }

    async fn connect_or_create(self, base_url: &str) -> Result<u64, ClientError> {
        match self {
            Self::Context => {
                let store = RemoteContextStore::connect_or_create(
                    base_url,
                    &CreateContextRequest {
                        name: "shared".into(),
                        storage_options: None,
                        id_index_type: None,
                        blob_columns: None,
                        embedding_dim: None,
                        distance_metric: None,
                    },
                )
                .await?;
                Ok(store.version())
            }
            Self::Rollout => {
                let store = RemoteRolloutStore::connect_or_create(
                    base_url,
                    &CreateRolloutStoreRequest {
                        name: "shared".into(),
                        storage_options: None,
                    },
                )
                .await?;
                Ok(store.version())
            }
            Self::Generic => {
                let store = RemoteGenericStore::connect_or_create(
                    base_url,
                    &CreateGenericStoreRequest {
                        name: "shared".into(),
                        schema: schema("requested"),
                        storage_options: None,
                        seal_on_add: false,
                    },
                )
                .await?;
                assert_eq!(store.spec(), &schema("stored"));
                Ok(store.version())
            }
            Self::Datagen => {
                let store = RemoteDatagenStore::connect_or_create(
                    base_url,
                    &CreateDatagenStoreRequest {
                        name: "shared".into(),
                        storage_options: None,
                    },
                )
                .await?;
                Ok(store.version())
            }
        }
    }
}

type Reply = (StatusCode, Json<Value>);

fn info(status: StatusCode, version: u64) -> Reply {
    (
        status,
        Json(json!({
            "name": "shared",
            "uri": "memory://shared",
            "version": version,
            "schema": schema("stored"),
        })),
    )
}

fn error(status: StatusCode, code: &str) -> Reply {
    (
        status,
        Json(json!({"error": {"code": code, "message": "server error"}})),
    )
}

struct MockState {
    gets: Vec<Reply>,
    posts: Vec<Reply>,
    get_count: AtomicUsize,
    post_count: AtomicUsize,
    initial_get_barrier: Option<Barrier>,
}

async fn get_store(State(state): State<Arc<MockState>>) -> Reply {
    let request = state.get_count.fetch_add(1, Ordering::SeqCst);
    if request < 2 {
        if let Some(barrier) = &state.initial_get_barrier {
            barrier.wait().await;
        }
    }
    state.gets[request].clone()
}

async fn create_store(State(state): State<Arc<MockState>>, Json(request): Json<Value>) -> Reply {
    assert_eq!(request["name"], "shared");
    let request = state.post_count.fetch_add(1, Ordering::SeqCst);
    state.posts[request].clone()
}

struct MockServer {
    base_url: String,
    state: Arc<MockState>,
    task: JoinHandle<()>,
}

impl MockServer {
    async fn start(
        kind: StoreKind,
        gets: Vec<Reply>,
        posts: Vec<Reply>,
        synchronize_initial_gets: bool,
    ) -> Self {
        let state = Arc::new(MockState {
            gets,
            posts,
            get_count: AtomicUsize::new(0),
            post_count: AtomicUsize::new(0),
            initial_get_barrier: synchronize_initial_gets.then(|| Barrier::new(2)),
        });
        let app = Router::new()
            .route(kind.path(), post(create_store))
            .route(&format!("{}/shared", kind.path()), get(get_store))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            base_url,
            state,
            task,
        }
    }

    fn assert_requests(&self, gets: usize, posts: usize) {
        assert_eq!(self.state.get_count.load(Ordering::SeqCst), gets);
        assert_eq!(self.state.post_count.load(Ordering::SeqCst), posts);
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test]
async fn concurrent_connect_or_create_reopens_the_winning_store() {
    for kind in STORE_KINDS {
        let server = MockServer::start(
            kind,
            vec![
                error(StatusCode::NOT_FOUND, "NOT_FOUND"),
                error(StatusCode::NOT_FOUND, "NOT_FOUND"),
                info(StatusCode::OK, 17),
            ],
            vec![
                info(StatusCode::CREATED, 17),
                error(StatusCode::CONFLICT, "ALREADY_EXISTS"),
            ],
            true,
        )
        .await;

        // Both lookups must see 404 before either caller can create the store.
        let (first, second) = timeout(Duration::from_secs(5), async {
            tokio::join!(
                kind.connect_or_create(&server.base_url),
                kind.connect_or_create(&server.base_url),
            )
        })
        .await
        .expect("concurrent connect_or_create timed out");

        assert_eq!(first.unwrap(), 17, "{kind:?}");
        assert_eq!(second.unwrap(), 17, "{kind:?}");
        server.assert_requests(3, 2);
    }
}

#[tokio::test]
async fn existing_store_does_not_attempt_creation() {
    for kind in STORE_KINDS {
        let server = MockServer::start(kind, vec![info(StatusCode::OK, 23)], vec![], false).await;

        assert_eq!(kind.connect_or_create(&server.base_url).await.unwrap(), 23);
        server.assert_requests(1, 0);
    }
}

#[tokio::test]
async fn unrelated_create_errors_are_returned_without_reopening() {
    for kind in STORE_KINDS {
        for (status, code) in [
            (StatusCode::CONFLICT, "COMPACTION_IN_PROGRESS"),
            (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL"),
        ] {
            let server = MockServer::start(
                kind,
                vec![error(StatusCode::NOT_FOUND, "NOT_FOUND")],
                vec![error(status, code)],
                false,
            )
            .await;

            let err = kind.connect_or_create(&server.base_url).await.unwrap_err();
            assert_api_error(err, status, code);
            server.assert_requests(1, 1);
        }
    }
}

#[tokio::test]
async fn reopen_errors_are_returned_after_a_create_conflict() {
    for kind in STORE_KINDS {
        let server = MockServer::start(
            kind,
            vec![
                error(StatusCode::NOT_FOUND, "NOT_FOUND"),
                error(StatusCode::SERVICE_UNAVAILABLE, "UNAVAILABLE"),
            ],
            vec![error(StatusCode::CONFLICT, "ALREADY_EXISTS")],
            false,
        )
        .await;

        let err = kind.connect_or_create(&server.base_url).await.unwrap_err();
        assert_api_error(err, StatusCode::SERVICE_UNAVAILABLE, "UNAVAILABLE");
        server.assert_requests(2, 1);
    }
}

fn assert_api_error(err: ClientError, expected_status: StatusCode, expected_code: &str) {
    match err {
        ClientError::Api {
            status,
            code,
            message,
        } => {
            assert_eq!(status, expected_status.as_u16());
            assert_eq!(code, expected_code);
            assert_eq!(message, "server error");
        }
        other => panic!("expected API error, got {other:?}"),
    }
}
