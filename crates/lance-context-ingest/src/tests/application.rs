use super::*;
use crate::{
    application::{self, config, Registry},
    http::{Accepted, Opened},
};
use futures::TryStreamExt;
use std::{
    path::Path,
    process::{Child, Command},
};

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
impl Process {
    fn spawn(binary: &str, config: &Path, role: &str, log: &Path) -> Result<Self> {
        let stderr = std::fs::File::create(log)?;
        Ok(Self(
            Command::new(binary)
                .arg(role)
                .arg(config)
                .env("TOKIO_WORKER_THREADS", "4")
                .stdout(stderr.try_clone()?)
                .stderr(stderr)
                .spawn()?,
        ))
    }
    async fn wait_success(&mut self) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if let Some(status) = self.0.try_wait()? {
                    ensure!(status.success(), "application subprocess failed: {status}");
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await?
    }
    async fn stop(&mut self) -> Result<()> {
        ensure!(
            Command::new("kill")
                .args(["-TERM", &self.0.id().to_string()])
                .status()?
                .success(),
            "signal failed"
        );
        self.wait_success().await
    }
}
async fn ready(
    client: &reqwest::Client,
    base: &str,
    token: &str,
    process: &mut Process,
) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            ensure!(
                process.0.try_wait()?.is_none(),
                "ingress exited before readiness"
            );
            if let Ok(response) = client
                .get(format!("{base}/api/v1/health"))
                .bearer_auth(token)
                .send()
                .await
            {
                if response.status().is_success() {
                    return Ok::<_, anyhow::Error>(());
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await?
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires INGEST_TEST_BIN and ETCD_TEST_ENDPOINTS; executes real application processes"]
async fn executable_bootstrap_http_restart_workers_and_managed_reader() -> Result<()> {
    let binary = std::env::var("INGEST_TEST_BIN")?;
    let endpoints = std::env::var("ETCD_TEST_ENDPOINTS")?;
    let root = tempfile::Builder::new()
        .prefix("application-")
        .tempdir_in(std::env::var("INGEST_TEST_ARTIFACTS")?)?
        .keep();
    let cluster = network::Cluster::open_signed(root.as_path(), layout()).await?;
    let mut config = cluster.application.clone();
    let path = root.as_path().join("managed.lance");
    config.tables.push(config::Table {
        binding: binding(),
        uri: path.to_str().unwrap().into(),
        storage_options: Default::default(),
        bootstrap: bootstrap::identity(),
    });
    let prefix = format!("/lance-context-test/ingest-application/{}", Uuid::new_v4());
    config.etcd = Some(config::Etcd {
        endpoints: endpoints.split(',').map(str::to_owned).collect(),
        prefix: prefix.clone(),
        ttl_seconds: 10,
        timeout: Duration::from_secs(3),
        tls: None,
    });
    let reserve = std::net::TcpListener::bind("127.0.0.1:0")?;
    let address = reserve.local_addr()?;
    let token = "application-test-authorization-token-0001";
    let bearer = root.as_path().join("token");
    std::fs::write(&bearer, token)?;
    let receipt = root.as_path().join("receipts.json");
    std::fs::write(
        &receipt,
        serde_json::to_vec(&serde_json::json!({"active": 1, "keys": {"1": vec![0x42u8;32]}}))?,
    )?;
    config.http = Some(config::Http {
        listen: address,
        bearer_token_file: bearer,
        receipt_key_file: receipt,
        options: Default::default(),
    });
    let file = root.as_path().join("config.json");
    std::fs::write(&file, serde_json::to_vec(&config)?)?;
    // Both invocations exercise config parsing, PEM loading and real TLS RPCs.
    for i in 0..2 {
        Process::spawn(
            &binary,
            &file,
            "bootstrap",
            &root.as_path().join(format!("bootstrap-{i}.log")),
        )?
        .wait_success()
        .await?;
    }
    let registry = Registry::connect(&application::Config::read(&file)?).await?;
    assert_eq!(
        registry
            .read("rollouts", None)
            .await?
            .count_rows(None)
            .await?,
        0
    );
    drop(reserve);
    let mut ingress = Process::spawn(
        &binary,
        &file,
        "ingress",
        &root.as_path().join("ingress.log"),
    )?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let base = format!("http://{address}");
    ready(&client, &base, token, &mut ingress).await?;
    assert_eq!(
        client
            .get(format!("{base}/api/v1/health"))
            .send()
            .await?
            .status(),
        401
    );
    let open = ProducerOpen {
        bucket: 0,
        slot: 0,
        expected_generation: 0,
        operation: Uuid::new_v4(),
    };
    let producer: Opened = client
        .post(format!("{base}/api/v1/ingest/rollouts/producer"))
        .bearer_auth(token)
        .json(&open)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let expected = batch(&["process-append"], &[&session(0)])?;
    let body = ipc_file(std::slice::from_ref(&expected))?;
    let append = || {
        client
            .post(format!("{base}/api/v1/ingest/rollouts/append"))
            .bearer_auth(token)
            .header(reqwest::header::CONTENT_TYPE, crate::http::ARROW_FILE)
            .header(crate::http::PRODUCER, &producer.producer)
            .header(crate::http::SEQUENCE, "1")
            .body(body.clone())
    };
    let accepted: Accepted = append().send().await?.error_for_status()?.json().await?;
    ingress.0.kill()?;
    ingress.0.wait()?;
    ingress = Process::spawn(
        &binary,
        &file,
        "ingress",
        &root.as_path().join("ingress-restart.log"),
    )?;
    ready(&client, &base, token, &mut ingress).await?;
    let retry: Accepted = append().send().await?.error_for_status()?.json().await?;
    assert_eq!(
        retry, accepted,
        "HTTP process restart changed exact retry receipt"
    );
    let mut workers = [
        Process::spawn(
            &binary,
            &file,
            "worker",
            &root.as_path().join("worker-1.log"),
        )?,
        Process::spawn(
            &binary,
            &file,
            "worker",
            &root.as_path().join("worker-2.log"),
        )?,
    ];
    let version = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            for (index, worker) in workers.iter_mut().enumerate() {
                if let Some(status) = worker.0.try_wait()? {
                    bail!(
                        "worker {} exited before publication: {status}; logs in {}",
                        index + 1,
                        root.display()
                    );
                }
            }
            let visible: crate::http::Visibility = client
                .post(format!("{base}/api/v1/ingest/rollouts/publication"))
                .bearer_auth(token)
                .json(&crate::http::ReceiptRequest {
                    receipt: accepted.receipt.clone(),
                })
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            if let Some(version) = visible.minimum_read_version {
                return Ok::<_, anyhow::Error>(version);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .with_context(|| format!("publication deadline; worker logs in {}", root.display()))??;
    for worker in &mut workers {
        worker.stop().await?;
    }
    ingress.stop().await?;
    let dataset = registry.read("rollouts", Some(version)).await?;
    assert_eq!(dataset.count_rows(None).await?, 1);
    let batches = dataset
        .scan()
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0], expected);
    assert!(registry.read("unauthorized", None).await.is_err());
    // Worker leases are revoked by joined shutdown. Cleanup is restricted to
    // this test's unique namespace; no shared service is restarted.
    let mut etcd =
        etcd_client::Client::connect(endpoints.split(',').collect::<Vec<_>>(), None).await?;
    assert!(etcd
        .get(
            prefix.clone(),
            Some(etcd_client::GetOptions::new().with_prefix())
        )
        .await?
        .kvs()
        .is_empty());
    drop((dataset, registry, workers, ingress));
    cluster.shutdown().await
}
