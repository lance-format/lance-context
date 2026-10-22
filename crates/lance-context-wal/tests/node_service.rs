use anyhow::{ensure, Context, Result};
use bytes::Bytes;
use lance_context_wal::{
    record::Batch,
    replication::{
        producer,
        runtime::{
            client::{self, Client, Reply, Request},
            Command,
        },
        Applied, Outcome,
    },
    service::{Config, Service},
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use uuid::Uuid;
#[path = "node_service/fixture.rs"]
mod fixture;
use fixture::Fixture;

#[test]
fn configuration_rejects_missing_signers_unknown_fields_and_listener_aliases() -> Result<()> {
    let f = Fixture::new()?;
    let original = &f.configs[&1];
    let mut bad = original.clone();
    bad.limits.producer_slots = 0;
    assert!(bad.validate().is_err());
    let mut bad = original.clone();
    bad.signing.sources.clear();
    assert!(bad.validate().is_err());
    let mut bad = f.configs[&4].clone();
    bad.signing.catalog = original.signing.catalog.clone();
    assert!(bad.validate().is_err());
    let mut bad = original.clone();
    bad.listen.snapshot = bad.listen.peer;
    assert!(bad.validate().is_err());
    let mut bad = original.clone();
    bad.endpoints.get_mut(&1).unwrap().peer = "foreign:7000".into();
    assert!(bad.validate().is_err());
    let mut json = serde_json::to_value(original)?;
    json["ignored_typo"] = true.into();
    assert!(serde_json::from_value::<Config>(json).is_err());
    for node in 1..=6 {
        assert_eq!(
            Config::read(&f.root.path().join(format!("{node}.json")))?.node,
            node
        );
    }
    Ok(())
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mismatched_tls_and_signing_identity_fail_before_directory_creation() -> Result<()> {
    let f = Fixture::new()?;
    let mut bad = f.configs[&1].clone();
    bad.tls.private_key = f.configs[&2].tls.private_key.clone();
    assert!(Service::start(bad).await.is_err());
    let mut bad = f.configs[&1].clone();
    bad.signing
        .sources
        .insert(10, f.root.path().join("source-20.pk8"));
    assert!(Service::start(bad).await.is_err());
    assert!(!f.configs[&1].directory.exists());
    Ok(())
}
async fn accepted(client: &Client, group: u64, batch: &Batch) -> Result<u64> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            // Explicitly retry transport errors with the same tuple and bytes.
            if let Ok(response) = client
                .request_routed(Request::Append {
                    group,
                    batch: batch.clone(),
                })
                .await
            {
                match response.reply {
                    Reply::Commit(Outcome::Committed(Applied::Accepted { index })) => {
                        return Ok(index)
                    }
                    Reply::Commit(Outcome::Unknown) | Reply::Unknown | Reply::Redirect { .. } => {}
                    other => anyhow::bail!("unexpected append reply: {other:?}"),
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?
}
async fn leader(f: &Fixture, group: u64) -> Result<u64> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            for &node in &f.layout.groups[&group] {
                if let Ok(status) = f.status(node).await {
                    let leader = status["groups"][group.to_string()]["status"]["leader"]
                        .as_u64()
                        .unwrap_or(0);
                    if leader != 0 && status["ready"] == true {
                        return Ok(leader);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn actual_node_binaries_maintain_two_disjoint_groups_catch_up_and_restart() -> Result<()> {
    let f = Fixture::new()?;
    let result = run_cluster(&f).await;
    if result.is_err() {
        let path = f.root.keep();
        eprintln!("failed service fixture retained at {}", path.display());
    }
    result
}
async fn run_cluster(f: &Fixture) -> Result<()> {
    let mut children = (1..=6)
        .map(|node| Ok((node, f.spawn(node, 1)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let first_leader = leader(f, 10).await?;
    leader(f, 20).await?;
    for child in children.values_mut() {
        child.assert_live()?;
    }
    let table = Uuid::from_u128(50);
    let producer = producer::producer_id(0, 1);
    for (group, bucket) in [(10, 0), (20, 1)] {
        let command = Command::Producer(producer::Command {
            cluster: f.layout.cluster,
            layout: f.layout.fingerprint()?,
            group,
            table,
            slot: 0,
            action: producer::Action::Open {
                expected_generation: 0,
                operation: Uuid::from_u128(group.into()),
                bucket,
            },
        });
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if let Ok(reply) = f
                    .client
                    .request_routed(Request::Commit(Box::new(command.clone())))
                    .await
                {
                    if matches!(
                        reply.reply,
                        Reply::Commit(Outcome::Committed(Applied::Producer(
                            producer::Reply::Opened { .. }
                        )))
                    ) {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await?;
    }
    let lagging = *(1..=3).find(|n| *n != first_leader).as_ref().unwrap();
    children.get_mut(&lagging).unwrap().kill()?;
    let mut expected = BTreeMap::new();
    let mut tails = BTreeMap::new();
    for sequence in 1..=16 {
        for (group, bucket) in [(10, 0), (20, 1)] {
            let batch = Batch::new(
                table,
                bucket,
                producer,
                sequence,
                Bytes::from(vec![(group + sequence) as u8; 256 << 10]),
            );
            let index = accepted(&f.client, group, &batch).await?;
            tails.insert(group, (index, batch.clone()));
            expected.insert((group, index), batch);
        }
    }
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let status = f.status(first_leader).await?;
            if status["groups"]["10"]["checkpoint_index"]
                .as_u64()
                .unwrap_or(0)
                >= tails[&10].0
                && status["maintenance"]["log_passes"].as_u64().unwrap_or(0) > 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    children.insert(lagging, f.spawn(lagging, 2)?);
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Ok(status) = f.status(lagging).await {
                if status["snapshots_installed"].as_u64().unwrap_or(0) > 0
                    && status["groups"]["10"]["status"]["accepted_records"] == 16
                {
                    break;
                }
            }
            children.get_mut(&lagging).unwrap().assert_live()?;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    children.get_mut(&first_leader).unwrap().kill()?;
    let batch = Batch::new(table, 0, producer, 17, Bytes::from(vec![0xac; 256 << 10]));
    let index = accepted(&f.client, 10, &batch).await?;
    tails.insert(10, (index, batch.clone()));
    expected.insert((10, index), batch);
    children.insert(first_leader, f.spawn(first_leader, 2)?);
    // Allow all replicas to catch up, then exercise SIGTERM during automatic
    // checkpoint/GC activity. Every new process reuses its original directory.
    for (&(group, index), batch) in &expected {
        for &node in &f.layout.groups[&group] {
            tokio::time::timeout(Duration::from_secs(30), async {
                loop {
                    if let Ok(reply) = f
                        .client
                        .request(node, Request::Fetch { group, index })
                        .await
                    {
                        if let Reply::Record(Some(meta)) = &reply.reply {
                            assert_eq!(meta, &batch.meta);
                            assert_eq!(reply.body, batch.body);
                            break;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await?;
        }
    }
    for child in children.values_mut() {
        child.terminate().await?;
    }
    for node in 1..=6 {
        children.insert(node, f.spawn(node, 3)?);
    }
    leader(f, 10).await?;
    leader(f, 20).await?;
    for (&group, (index, batch)) in &tails {
        assert_eq!(accepted(&f.client, group, batch).await?, *index);
    }
    for (&(group, index), batch) in &expected {
        for &node in &f.layout.groups[&group] {
            let reply = f
                .client
                .request(node, Request::Fetch { group, index })
                .await?;
            ensure!(
                matches!(&reply.reply,Reply::Record(Some(meta)) if meta==&batch.meta),
                "record metadata lost across restart"
            );
            assert_eq!(reply.body, batch.body);
        }
    }
    for child in children.values_mut() {
        child.terminate().await?;
    }
    Ok(())
}
