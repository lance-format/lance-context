//! Keep the original writers running across voter loss. Unknown appends are
//! resolved by quorum reads after recovery, never by resubmitting their bodies.
use super::*;
use lance_context_wal::replication::query::{self, Query, Value};
use std::time::Instant;
use tokio::{sync::mpsc, task::JoinSet};

const TABLE: Uuid = Uuid::from_u128(51);

struct Attempt {
    group: u64,
    began: Instant,
    batch: Batch,
    outcome: std::result::Result<u64, String>,
}

async fn append_once(
    client: &Client,
    group: u64,
    batch: &Batch,
) -> Result<std::result::Result<u64, String>> {
    let response = match client
        .request_routed(Request::Append {
            group,
            batch: batch.clone(),
        })
        .await
    {
        Ok(response) => response,
        Err(error) => return Ok(Err(format!("{error:#}"))),
    };
    match response.reply {
        Reply::Commit(Outcome::Committed(Applied::Accepted { index })) => Ok(Ok(index)),
        Reply::Unknown | Reply::Commit(Outcome::Unknown) => {
            Ok(Err(format!("{:?}", response.reply)))
        }
        other => anyhow::bail!(
            "unexpected refusal or protocol reply, not quorum-loss evidence: {other:?}"
        ),
    }
}

async fn writer(
    client: Arc<Client>,
    group: u64,
    bucket: u32,
    tx: mpsc::Sender<Attempt>,
) -> Result<()> {
    // The channel bounds evidence and prevents a slow test controller from
    // generating an unbounded load. Every lane has only one outstanding RPC.
    for sequence in 1..=1024 {
        let batch = Batch::new(
            TABLE,
            bucket,
            producer::producer_id(0, 1),
            sequence,
            Bytes::from(vec![(group + sequence) as u8; 16 << 10]),
        );
        let began = Instant::now();
        let outcome = append_once(&client, group, &batch).await?;
        let unknown = outcome.is_err();
        tx.send(Attempt {
            group,
            began,
            batch,
            outcome,
        })
        .await
        .map_err(|_| anyhow::anyhow!("attempt observer closed"))?;
        if unknown {
            return Ok(());
        }
    }
    anyhow::bail!("writer exhausted its bounded input before observing quorum loss")
}

async fn receive(rx: &mut mpsc::Receiver<Attempt>) -> Result<Attempt> {
    tokio::time::timeout(Duration::from_secs(30), rx.recv())
        .await?
        .context("original writers ended before the fault boundary")
}

async fn tail(
    f: &Fixture,
    group: u64,
    batch: &Batch,
) -> Result<Option<(lance_context_wal::record::BatchMeta, u64)>> {
    // Only read-index queries are repeated during the recovery election.
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Ok(response) = f
                .client
                .request_routed(Request::Query {
                    group,
                    query: Query::Producer {
                        table: batch.meta.stream,
                        bucket: batch.meta.bucket,
                        producer: batch.meta.producer,
                    },
                })
                .await
            {
                if let Reply::Query(query::Outcome::Observed {
                    cut,
                    value: Value::Producer(value),
                }) = response.reply
                {
                    ensure!(
                        cut.applied >= cut.quorum_index,
                        "unapplied read-index result"
                    );
                    return Ok(value);
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batched_peers_preserve_live_writers_across_voter_and_quorum_loss() -> Result<()> {
    let mut f = Fixture::shared_groups(true)?;
    for (&node, config) in &mut f.configs {
        config.peer.send_batch_frames = 32;
        config.peer.send_batch_bytes = 16 << 20;
        config.validate()?;
        std::fs::write(
            f.root.path().join(format!("{node}.json")),
            serde_json::to_vec(config)?,
        )?;
    }
    let result = run(&f).await;
    if result.is_err() {
        eprintln!(
            "failed live-quorum fixture retained at {}",
            f.root.keep().display()
        );
    }
    result
}

async fn run(f: &Fixture) -> Result<()> {
    let mut children = (1..=3)
        .map(|id| Ok((id, f.spawn(id, 1)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let a = leader(f, 10).await?;
    let b = leader(f, 11).await?;
    leader(f, 5).await?;
    shared_groups::open(f, 10, 0, TABLE).await?;
    shared_groups::open(f, 11, 2, TABLE).await?;
    shared_groups::open_slot(f, 10, 0, TABLE, 1).await?;
    // The first killed process is a follower of both data groups. Losing a
    // leader is covered separately by the existing process-relocation test.
    let follower = (1..=3).find(|id| *id != a && *id != b).unwrap();
    let second = (1..=3).find(|id| *id != a && *id != follower).unwrap();
    let (tx, mut rx) = mpsc::channel(8);
    let mut writers = JoinSet::new();
    for (group, bucket) in [(10, 0), (11, 2)] {
        writers.spawn(writer(f.client.clone(), group, bucket, tx.clone()));
    }
    drop(tx);
    let mut attempts = Vec::new();
    let mut before = BTreeMap::<u64, usize>::new();
    while before.values().sum::<usize>() < 8 || before.len() != 2 {
        let attempt = receive(&mut rx).await?;
        ensure!(
            attempt.outcome.is_ok(),
            "writer failed before voter loss: {:?}",
            attempt.outcome
        );
        *before.entry(attempt.group).or_default() += 1;
        attempts.push(attempt);
    }
    ensure!(
        writers.try_join_next().is_none(),
        "writer ended before voter loss"
    );
    children.get_mut(&follower).unwrap().kill()?;
    let one_lost = Instant::now();
    let mut during = BTreeMap::<u64, usize>::new();
    while ![10, 11]
        .iter()
        .all(|group| during.get(group).copied().unwrap_or(0) >= 4)
    {
        let attempt = receive(&mut rx).await?;
        ensure!(
            attempt.outcome.is_ok(),
            "single-voter loss interrupted a writer: {:?}",
            attempt.outcome
        );
        if attempt.began >= one_lost {
            *during.entry(attempt.group).or_default() += 1;
        }
        attempts.push(attempt);
    }
    ensure!(
        writers.try_join_next().is_none(),
        "writers settled before quorum loss"
    );
    children.get_mut(&second).unwrap().kill()?;
    let quorum_lost = Instant::now();
    // This producer has no earlier request: the append is definitely first
    // submitted after both killed children have exited, not just replied late.
    let probe = Batch::new(
        TABLE,
        0,
        producer::producer_id(1, 1),
        1,
        Bytes::from_static(b"isolated append"),
    );
    let isolated_append = f
        .client
        .request(
            a,
            Request::Append {
                group: 10,
                batch: probe.clone(),
            },
        )
        .await;
    ensure!(
        !matches!(&isolated_append, Ok(response) if matches!(response.reply, Reply::Commit(Outcome::Committed(Applied::Accepted { .. })))),
        "isolated voter acknowledged an append submitted after quorum loss"
    );
    while let Some(attempt) = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await? {
        // An ACK for a request begun before the kill can already have committed.
        if attempt.began >= quorum_lost {
            ensure!(
                attempt.outcome.is_err(),
                "isolated voter acknowledged a new request"
            );
        }
        attempts.push(attempt);
    }
    while let Some(result) = writers.join_next().await {
        result??;
    }
    let unresolved: Vec<_> = attempts.iter().filter(|x| x.outcome.is_err()).collect();
    ensure!(
        unresolved.len() == 2,
        "each original writer must preserve its first unknown request"
    );
    // A cached local tail must not masquerade as a quorum read either.
    let isolated = f
        .client
        .request(
            a,
            Request::Query {
                group: 10,
                query: Query::Producer {
                    table: TABLE,
                    bucket: 0,
                    producer: producer::producer_id(0, 1),
                },
            },
        )
        .await;
    ensure!(
        !matches!(isolated, Ok(ref response) if matches!(response.reply, Reply::Query(query::Outcome::Observed { .. }))),
        "isolated voter released a quorum read"
    );
    std::fs::write(
        f.root.path().join("live-fault-outcomes.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "single_voter_acks":during,"follower":follower,"second":second,
            "attempts":attempts.iter().map(|x| serde_json::json!({"group":x.group,"meta":x.batch.meta,"outcome":x.outcome})).collect::<Vec<_>>(),
            "isolated_append":format!("{:?}",isolated_append.as_ref().map(|response| &response.reply)),
            "mutation_resubmissions":0
        }))?,
    )?;

    // End the last original process too, so no old RPC handler can submit a
    // delayed request after the recovery read. Reopen all original stores;
    // nothing is resent to make recovery pass.
    children.get_mut(&a).unwrap().kill()?;
    for node in 1..=3 {
        children.insert(node, f.spawn(node, 2)?);
    }
    leader(f, 10).await?;
    leader(f, 11).await?;
    if let Some((meta, index)) = tail(f, 10, &probe).await? {
        ensure!(
            meta == probe.meta,
            "isolated request recovered with different input"
        );
        for node in 1..=3 {
            shared_groups::fetch(f, node, 10, index, &probe).await?;
        }
    }
    for group in [10, 11] {
        let unknown = unresolved.iter().find(|x| x.group == group).unwrap();
        let last = attempts
            .iter()
            .rev()
            .find(|x| x.group == group && x.outcome.is_ok())
            .unwrap();
        let (meta, index) = tail(f, group, &unknown.batch)
            .await?
            .context("acknowledged producer lost")?;
        ensure!(
            meta == last.batch.meta || meta == unknown.batch.meta,
            "recovered tail is outside original acknowledged/unknown prefix"
        );
        if meta == last.batch.meta {
            ensure!(
                index == *last.outcome.as_ref().unwrap(),
                "acknowledged index changed"
            );
        } else {
            for node in 1..=3 {
                shared_groups::fetch(f, node, group, index, &unknown.batch).await?;
            }
        }
    }
    for attempt in &attempts {
        if let Ok(index) = attempt.outcome {
            for node in 1..=3 {
                shared_groups::fetch(f, node, attempt.group, index, &attempt.batch).await?;
            }
        }
    }
    for child in children.values_mut() {
        child.terminate().await?;
    }
    Ok(())
}
