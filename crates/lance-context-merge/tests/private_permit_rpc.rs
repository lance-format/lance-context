//! Run with `python3 scripts/ci/test_commit_permits.py`; uses a private loopback etcd.
use etcd_client::Client;
use lance_context_merge::{target_lock_key, ClaimProof, Coordinator, Execution};
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio::{sync::Barrier, task::JoinSet};

fn metrics() -> BTreeMap<String, f64> {
    let endpoint = std::env::var("RK_PRIVATE_ETCD").unwrap();
    assert!(endpoint.starts_with("http://127.0.0.1:"));
    let out = std::process::Command::new("curl")
        .args([
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            "3",
            &format!("{endpoint}/metrics"),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    let mut counts = BTreeMap::new();
    for line in text
        .lines()
        .filter(|l| l.starts_with("grpc_server_started_total{"))
    {
        let (labels, value) = line.rsplit_once(' ').unwrap();
        let method = labels
            .split("grpc_method=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        *counts.entry(method.to_string()).or_insert(0.) += value.parse::<f64>().unwrap();
    }
    counts
}

fn phase(rows: &mut Vec<Value>, name: &str, before: BTreeMap<String, f64>, extra: Value) {
    let after = metrics();
    let expected_txns = match name {
        "serial_base_100" | "serial_shard_100" => Some(100.),
        "concurrent_200" => Some(200.),
        _ => None,
    };
    if let Some(expected) = expected_txns {
        assert_eq!(
            after["Range"] - before.get("Range").copied().unwrap_or(0.),
            0.
        );
        assert_eq!(
            after["Txn"] - before.get("Txn").copied().unwrap_or(0.),
            expected
        );
    }
    let delta: BTreeMap<_, _> = after
        .iter()
        .map(|(k, v)| (k, v - before.get(k).unwrap_or(&0.)))
        .collect();
    let row = json!({"phase":name,"rpc_delta":delta,"detail":extra});
    println!("{row}");
    rows.push(row);
}

async fn setup(
    client: &mut Client,
    coordinator: &Coordinator,
    prefix: &str,
    target: &str,
) -> (ClaimProof, Execution) {
    let proof = ClaimProof {
        key: format!("{prefix}/claims/{target}"),
        token: uuid::Uuid::new_v4().to_string(),
        lease_id: 0,
    };
    client
        .put(proof.key.clone(), proof.token.clone(), None)
        .await
        .unwrap();
    client
        .put(target_lock_key(prefix, target), proof.token.clone(), None)
        .await
        .unwrap();
    let reserved = Execution::new(target, "private-fixture", "private-fixture", 600);
    assert!(coordinator.reserve(&proof, &reserved).await.unwrap());
    let running = coordinator.start(&reserved).await.unwrap().unwrap();
    (proof, running)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires dedicated private loopback etcd; never use production"]
async fn private_permit_measure_permit_rpc_and_fence() {
    tokio::time::timeout(Duration::from_secs(90), run())
        .await
        .expect("finite fixture deadline");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires dedicated private loopback etcd and response-loss proxy"]
async fn private_permit_lost_ack_preserves_union_and_fence() {
    tokio::time::timeout(Duration::from_secs(45), async {
        let endpoint = std::env::var("RK_PRIVATE_ETCD").unwrap();
        let proxy = std::env::var("RK_PRIVATE_PROXY").unwrap();
        assert!(endpoint.starts_with("http://127.0.0.1:"));
        assert!(proxy.starts_with("http://127.0.0.1:"));
        let dir = std::path::PathBuf::from(std::env::var("RK_PRIVATE_PROXY_DIR").unwrap());
        let prefix = format!("/private-lost-ack/{}", uuid::Uuid::new_v4());
        let mut direct = Client::connect([endpoint], None).await.unwrap();
        let observer = Coordinator::new(direct.clone(), prefix.clone());
        let (proof, running) = setup(&mut direct, &observer, &prefix, "lost-ack").await;
        let proxied = Coordinator::new(Client::connect([proxy], None).await.unwrap(), prefix.clone());
        let publisher = Arc::new(proxied.commit_permit_publisher(&running, "memory://fixture"));
        publisher.authorize("base", 7).await.unwrap();
        // The publisher performs a CAS without a preparatory GET.
        std::fs::write(dir.join("drop-arm.json"), br#"{"data_frame":1}"#).unwrap();
        let before = metrics();
        let outcome = publisher.authorize("base", 8).await;
        assert!(outcome.is_err(), "lost acknowledgement must not authorize storage I/O");
        assert!(dir.join("drop-evidence.json").exists(), "proxy must have fired");
        let permit_key = format!("{prefix}/merge-commit-permits/{}", running.id);
        let row = direct.get(permit_key, None).await.unwrap();
        let value: Value = serde_json::from_slice(row.kvs()[0].value()).unwrap();
        assert_eq!(value["versions"]["base"], 8, "the unacknowledged transaction actually committed");
        let shard = format!("shard:{}", uuid::Uuid::new_v4());
        observer.authorize_commit(&running, "memory://fixture", &shard, 99).await.unwrap();
        publisher.authorize("base", 9).await.unwrap();
        // Hold a real committed transaction response, then cancel its caller.
        // Reusing this same publisher must reconcile its still-old revision.
        std::fs::rename(dir.join("drop-evidence.json"), dir.join("lost-ack-evidence.json")).unwrap();
        std::fs::write(dir.join("drop-arm.json"), br#"{"data_frame":1,"mode":"hold"}"#).unwrap();
        let cancelled = tokio::spawn({
            let publisher = publisher.clone();
            async move { publisher.authorize("base", 12).await }
        });
        while !dir.join("drop-evidence.json").exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let row = direct.get(format!("{prefix}/merge-commit-permits/{}", running.id), None).await.unwrap();
        let value: Value = serde_json::from_slice(row.kvs()[0].value()).unwrap();
        assert_eq!(value["versions"]["base"], 12);
        assert!(!cancelled.is_finished());
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());
        std::fs::write(dir.join("drop-release"), b"release").unwrap();
        observer.authorize_commit(&running, "memory://fixture", &shard, 102).await.unwrap();
        publisher.authorize("base", 13).await.unwrap();
        let frozen = observer.freeze(&proof, &running).await.unwrap().unwrap();
        let marks = observer.watermarks(&frozen).await.unwrap();
        assert_eq!(marks.versions["base"], 13);
        assert_eq!(marks.versions[&shard], 102);
        assert!(publisher.authorize("base", 14).await.is_err());
        let after = observer.watermarks(&frozen).await.unwrap();
        assert_eq!(marks.versions, after.versions);
        let mut rows = vec![];
        phase(&mut rows, "lost_ack_external_writer_retry_freeze", before, json!({"ack_error":outcome.unwrap_err(),"persisted_without_ack":8,"persisted_before_cancellation":12,"final_watermarks":marks,"post_freeze_success":0}));
        direct.delete(prefix, Some(etcd_client::DeleteOptions::new().with_prefix())).await.unwrap();
        std::fs::write(dir.join("lost-ack-result.json"), serde_json::to_vec_pretty(&json!({"status":"PASS","phases":rows})).unwrap()).unwrap();
    }).await.expect("finite lost-ack deadline");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires dedicated private loopback etcd"]
async fn private_permit_independent_publishers_preserve_out_of_order_union() {
    tokio::time::timeout(Duration::from_secs(45), async {
        let endpoint = std::env::var("RK_PRIVATE_ETCD").unwrap();
        assert!(endpoint.starts_with("http://127.0.0.1:"));
        let prefix = format!("/private-independent-publishers/{}", uuid::Uuid::new_v4());
        let mut client = Client::connect([endpoint.clone()], None).await.unwrap();
        let coordinator = Coordinator::new(client.clone(), prefix.clone());
        let (proof, running) = setup(&mut client, &coordinator, &prefix, "independent").await;
        let other = Coordinator::new(
            Client::connect([endpoint], None).await.unwrap(),
            prefix.clone(),
        );
        let first = coordinator.commit_permit_publisher(&running, "memory://fixture");
        let second = other.commit_permit_publisher(&running, "memory://fixture");
        let shards: Vec<_> = (0..2)
            .map(|_| format!("shard:{}", uuid::Uuid::new_v4()))
            .collect();
        first.authorize(&shards[0], 7).await.unwrap();
        second.authorize(&shards[1], 99).await.unwrap();
        let before = metrics();
        let (left, right) = tokio::join!(
            async {
                for v in [100, 2, 90, 1, 55] {
                    first.authorize("base", v).await?;
                    first.authorize(&shards[0], v).await?;
                }
                Ok::<_, String>(())
            },
            async {
                for v in [80, 3, 120, 5, 70] {
                    second.authorize("base", v).await?;
                    second.authorize(&shards[1], v).await?;
                }
                Ok::<_, String>(())
            }
        );
        left.unwrap();
        right.unwrap();
        let wrong = other.commit_permit_publisher(&running, "memory://wrong-dataset");
        assert!(wrong.authorize("base", 999).await.is_err());
        assert!(first.authorize("shard:invalid", 999).await.is_err());
        let frozen = coordinator.freeze(&proof, &running).await.unwrap().unwrap();
        let marks = coordinator.watermarks(&frozen).await.unwrap();
        assert_eq!(marks.versions["base"], 120);
        assert_eq!(marks.versions[&shards[0]], 100);
        assert_eq!(marks.versions[&shards[1]], 120);
        assert!(first.authorize("base", 1).await.is_err());
        assert!(second.authorize("base", 1).await.is_err());
        let mut rows = vec![];
        phase(
            &mut rows,
            "independent_publishers_out_of_order",
            before,
            json!({"watermarks":marks}),
        );
        client
            .delete(
                prefix,
                Some(etcd_client::DeleteOptions::new().with_prefix()),
            )
            .await
            .unwrap();
        let dir = std::path::PathBuf::from(std::env::var("RK_PRIVATE_PROXY_DIR").unwrap());
        std::fs::write(
            dir.join("independent-result.json"),
            serde_json::to_vec_pretty(&json!({"status":"PASS","phases":rows})).unwrap(),
        )
        .unwrap();
    })
    .await
    .expect("finite independent publisher deadline");
}

async fn run() {
    let endpoint = std::env::var("RK_PRIVATE_ETCD").unwrap();
    assert!(endpoint.starts_with("http://127.0.0.1:"));
    let prefix = format!("/private-permit-fixture/{}", uuid::Uuid::new_v4());
    let mut client = Client::connect([endpoint], None).await.unwrap();
    let coordinator = Coordinator::new(client.clone(), prefix.clone());
    let mut rows = vec![];
    let before = metrics();
    let (proof, running) = setup(&mut client, &coordinator, &prefix, "serial").await;
    phase(&mut rows, "setup_serial", before, json!({}));
    let publisher = coordinator.commit_permit_publisher(&running, "memory://fixture");
    let before = metrics();
    for v in 1..=100 {
        publisher.authorize("base", v).await.unwrap();
    }
    phase(&mut rows, "serial_base_100", before, json!({"success":100}));
    let shards: Vec<String> = (0..8)
        .map(|_| format!("shard:{}", uuid::Uuid::new_v4()))
        .collect();
    let before = metrics();
    for v in 1..=100 {
        publisher.authorize(&shards[0], v).await.unwrap();
    }
    phase(
        &mut rows,
        "serial_shard_100",
        before,
        json!({"success":100}),
    );
    let before = metrics();
    let frozen = coordinator.freeze(&proof, &running).await.unwrap().unwrap();
    let marks = coordinator.watermarks(&frozen).await.unwrap();
    assert_eq!(marks.versions["base"], 100);
    assert_eq!(marks.versions[&shards[0]], 100);
    phase(
        &mut rows,
        "serial_freeze_verify",
        before,
        json!({"watermarks":marks}),
    );

    for race in [false, true] {
        let before = metrics();
        let (proof, running) = setup(
            &mut client,
            &coordinator,
            &prefix,
            if race { "race" } else { "concurrent" },
        )
        .await;
        phase(
            &mut rows,
            if race {
                "setup_race"
            } else {
                "setup_concurrent"
            },
            before,
            json!({}),
        );
        let before = metrics();
        let publisher = Arc::new(coordinator.commit_permit_publisher(&running, "memory://fixture"));
        let barrier = Arc::new(Barrier::new(9));
        let mut tasks = JoinSet::new();
        for resource in &shards {
            let (publisher, b, resource) = (publisher.clone(), barrier.clone(), resource.clone());
            tasks.spawn(async move {
                b.wait().await;
                let mut admitted = Vec::new();
                let mut rejected = Vec::new();
                for version in 1..=25 {
                    match publisher.authorize(&resource, version).await {
                        Ok(()) => admitted.push(version),
                        Err(e) => rejected.push(e),
                    }
                }
                (resource, admitted, rejected)
            });
        }
        barrier.wait().await;
        let early_frozen = if race {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Some(coordinator.freeze(&proof, &running).await.unwrap().unwrap())
        } else {
            None
        };
        let mut outcomes = Vec::new();
        while let Some(row) = tasks.join_next().await {
            outcomes.push(row.unwrap());
        }
        let successes: usize = outcomes.iter().map(|(_, yes, _)| yes.len()).sum();
        let failures: usize = outcomes.iter().map(|(_, _, no)| no.len()).sum();
        if !race {
            assert_eq!(successes, 200);
            assert_eq!(failures, 0);
        }
        phase(
            &mut rows,
            if race {
                "concurrent_freeze_race"
            } else {
                "concurrent_200"
            },
            before,
            json!({"success":successes,"rejected":failures,"includes_freeze_txn":race,"outcomes":outcomes}),
        );
        let before = metrics();
        let frozen = match early_frozen {
            Some(f) => f,
            None => coordinator.freeze(&proof, &running).await.unwrap().unwrap(),
        };
        let marks = coordinator.watermarks(&frozen).await.unwrap();
        for (resource, yes, _) in &outcomes {
            if let Some(max) = yes.iter().max() {
                assert!(marks.versions[resource] >= *max);
            }
        }
        for resource in &shards {
            assert!(publisher.authorize(resource, 1000).await.is_err());
        }
        let after_marks = coordinator.watermarks(&frozen).await.unwrap();
        assert_eq!(marks.versions, after_marks.versions);
        phase(
            &mut rows,
            "verify_frozen_and_reject_8",
            before,
            json!({"watermarks":marks,"post_freeze_success":0}),
        );
    }
    let before = metrics();
    client
        .delete(
            prefix,
            Some(etcd_client::DeleteOptions::new().with_prefix()),
        )
        .await
        .unwrap();
    phase(&mut rows, "cleanup", before, json!({}));
    std::fs::write(
        std::env::var("RK_PRIVATE_RESULT").unwrap(),
        serde_json::to_vec_pretty(
            &json!({"phases":rows,"status":"PASS","fault_results":"lost-ack-result.json"}),
        )
        .unwrap(),
    )
    .unwrap();
}
