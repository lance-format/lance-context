//! Actual Azure SDK DELETE requests against real RF3/Lance orphan files.
use super::*;
use object_store::{
    azure::MicrosoftAzureBuilder,
    client::{
        HttpClient, HttpConnector, HttpError, HttpRequest, HttpResponse, HttpResponseBody,
        HttpService,
    },
    local::LocalFileSystem,
    path::Path,
    ClientOptions, ObjectStoreExt, RetryConfig,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use tokio::sync::Semaphore;

#[derive(Debug, Default)]
struct State {
    targets: Vec<Path>,
    lost_batch_reply: bool,
    bulk: Vec<Vec<Path>>,
    singles: Vec<(Path, u16)>,
}

#[derive(Clone, Debug)]
struct AzureDelete {
    state: Arc<Mutex<State>>,
    gate: Arc<Semaphore>,
    inflight: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl HttpConnector for AzureDelete {
    fn connect(&self, _: &ClientOptions) -> object_store::Result<HttpClient> {
        Ok(HttpClient::new(self.clone()))
    }
}

fn path(uri: &str) -> Path {
    let url = url::Url::parse(&format!("https://account.blob.core.windows.net{uri}")).unwrap();
    let relative = url.path().strip_prefix("/container/").unwrap();
    Path::from_url_path(relative).unwrap()
}

fn response(status: u16, body: String, multipart: bool) -> HttpResponse {
    let mut response = HttpResponse::new(HttpResponseBody::from(body));
    *response.status_mut() = status.try_into().unwrap();
    if multipart {
        response.headers_mut().insert(
            "content-type",
            "multipart/mixed; boundary=response".parse().unwrap(),
        );
    }
    response
}

async fn remove(path: &Path) -> u16 {
    match LocalFileSystem::new().delete(path).await {
        Ok(()) => 202,
        Err(object_store::Error::NotFound { .. }) => 404,
        Err(error) => panic!("unexpected local deletion failure: {error}"),
    }
}

#[async_trait::async_trait]
impl HttpService for AzureDelete {
    async fn call(&self, request: HttpRequest) -> std::result::Result<HttpResponse, HttpError> {
        assert_eq!(request.headers()["authorization"], "Bearer test-token");
        if request.method().as_str() == "DELETE" {
            assert!(request.uri().query().is_none());
            let path = path(request.uri().path());
            let count = self.inflight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(count, Ordering::SeqCst);
            let permit = self.gate.acquire().await.unwrap();
            let status = remove(&path).await;
            self.state.lock().unwrap().singles.push((path, status));
            self.inflight.fetch_sub(1, Ordering::SeqCst);
            drop(permit);
            return Ok(response(status, String::new(), false));
        }
        assert_eq!(request.method().as_str(), "POST");
        assert!(request
            .uri()
            .query()
            .unwrap()
            .split('&')
            .any(|p| p == "comp=batch"));
        let body = HttpResponseBody::new(request.into_body()).bytes().await?;
        let paths: Vec<_> = std::str::from_utf8(&body)
            .unwrap()
            .lines()
            .filter_map(|line| {
                line.strip_prefix("DELETE ")
                    .and_then(|v| v.strip_suffix(" HTTP/1.1"))
            })
            .map(path)
            .collect();
        assert!(paths.len() > 1, "singleton must bypass Blob Batch");
        let (targets, lost) = {
            let mut state = self.state.lock().unwrap();
            state.bulk.push(paths.clone());
            (state.targets.clone(), state.lost_batch_reply)
        };
        if lost && paths.iter().any(|p| targets.contains(p)) {
            // The request may have removed some files before its reply failed.
            remove(&targets[0]).await;
            return Ok(response(
                503,
                "<Error><Code>ServerBusy</Code></Error>".into(),
                false,
            ));
        }
        let mut reply = String::new();
        for (index, path) in paths.iter().enumerate() {
            let status = if targets.get(1) == Some(path) {
                503
            } else {
                remove(path).await
            };
            let reason = match status {
                202 => "Accepted",
                404 => "Not Found",
                _ => "Service Unavailable",
            };
            reply.push_str(&format!("--response\r\nContent-Type: application/http\r\nContent-ID: {index}\r\n\r\nHTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\n\r\n\r\n"));
        }
        reply.push_str("--response--\r\n");
        Ok(response(202, reply, true))
    }
}

#[tokio::test]
async fn failed_azure_bulk_deletion_falls_back_and_drains_real_lance() -> Result<()> {
    for lost_batch_reply in [false, true] {
        let root = root()?;
        let directory = root.path().join("turns.lance");
        let c =
            network::Cluster::open_signed(root.path(), annotation_publication::one_shard()).await?;
        let transport = AzureDelete {
            state: Arc::new(Mutex::new(State {
                lost_batch_reply,
                ..Default::default()
            })),
            gate: Arc::new(Semaphore::new(0)),
            inflight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
        };
        let sdk = MicrosoftAzureBuilder::new()
            .with_account("account")
            .with_container_name("container")
            .with_bearer_token_authorization("test-token")
            .with_retry(RetryConfig {
                max_retries: 0,
                ..Default::default()
            })
            .with_http_connector(transport.clone())
            .build()?;
        let mut faults = Faults::default();
        faults.azure_delete = Some(Arc::new(sdk));
        let storage = fault_store::storage(&directory, Arc::new(faults)).await?;
        let (first, original, abandoned) = acknowledged_orphans(&c, &storage, &directory).await?;
        let targets = abandoned
            .iter()
            .map(Path::from_absolute_path)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        transport.state.lock().unwrap().targets = targets.clone();
        let (service, runtime) = owned(&c, &storage, &limits()).await?;
        let (stop, task) = run_owner(whole_page(runtime)?);
        let expected = if lost_batch_reply { 2 } else { 1 };
        let check = async {
            tokio::time::timeout(Duration::from_secs(30), async {
                while transport.inflight.load(Ordering::SeqCst) < expected {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await?;
            let observed = service.reclamation_observation();
            ensure!(
                observed.pending_objects == expected && observed.delete_failures >= expected as u64
            );
            ensure!(transport.peak.load(Ordering::SeqCst) <= 2);
            let value = service.get("T", Duration::from_secs(5)).await?.unwrap();
            ensure!(value.status.as_deref() == Some("acknowledged"));
            {
                let state = transport.state.lock().unwrap();
                ensure!(state.bulk.len() == 1 && state.bulk[0].len() == 2);
                ensure!(state.singles.is_empty(), "held deletes completed early");
            }
            transport.gate.add_permits(2);
            tokio::time::timeout(Duration::from_secs(30), async {
                while abandoned.iter().any(|p| p.exists())
                    || service.reclamation_observation().pending_objects != 0
                {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await?;
            {
                let state = transport.state.lock().unwrap();
                let retried: BTreeSet<_> = state
                    .singles
                    .iter()
                    .filter(|(p, _)| targets.contains(p))
                    .map(|(p, _)| p.clone())
                    .collect();
                let required: BTreeSet<_> = targets
                    .iter()
                    .skip(usize::from(!lost_batch_reply))
                    .cloned()
                    .collect();
                ensure!(retried == required, "retry changed the unresolved path set");
                if lost_batch_reply {
                    ensure!(state.singles.contains(&(targets[0].clone(), 404)));
                }
            }
            ensure!(service.submit(&first, Duration::from_secs(30)).await? == original);
            let producer = producer(&service, 2).await?;
            for sequence in 1..=4 {
                let update = request(
                    &service,
                    &producer,
                    sequence,
                    wire(&limits(), &["T"], 0, None, 1, Some(sequence as f32))?,
                )
                .await?;
                let completed = service.submit(&update, Duration::from_secs(30)).await?;
                ensure!(completed.result == MutationResult::Applied);
                let value = service.get("T", Duration::from_secs(5)).await?.unwrap();
                ensure!(
                    value.status.as_deref() == Some("acknowledged")
                        && value.score == Some(sequence as f32)
                );
                drained(&c, &service, completed.index).await?;
            }
            Ok::<_, anyhow::Error>(())
        }
        .await;
        transport.gate.add_permits(2);
        let _ = stop.send(());
        let stopped = task.await?;
        let cold = if check.is_ok() {
            cold_check(&c, &storage, Some(4.0)).await
        } else {
            Ok(())
        };
        c.shutdown().await?;
        check?;
        stopped?;
        cold?;
    }
    Ok(())
}
