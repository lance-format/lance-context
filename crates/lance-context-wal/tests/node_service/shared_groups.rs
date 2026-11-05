use super::*;

async fn open(f: &Fixture, group: u64, bucket: u32, table: Uuid) -> Result<()> {
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
            if let Ok(response) = f
                .client
                .request_routed(Request::Commit(Box::new(command.clone())))
                .await
            {
                if matches!(
                    response.reply,
                    Reply::Commit(Outcome::Committed(Applied::Producer(
                        producer::Reply::Opened { .. }
                    )))
                ) {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?;
    Ok(())
}

async fn fetch(f: &Fixture, node: u64, group: u64, index: u64, batch: &Batch) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Ok(response) = f
                .client
                .request(node, Request::Fetch { group, index })
                .await
            {
                if let Reply::Record(Some(meta)) = response.reply {
                    assert_eq!(meta, batch.meta);
                    assert_eq!(response.body, batch.body);
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shared_engine_data_groups_catch_up_and_survive_full_process_crash() -> Result<()> {
    let f = Fixture::shared_groups()?;
    let result = run(&f).await;
    if result.is_err() {
        eprintln!(
            "failed shared-group fixture retained at {}",
            f.root.keep().display()
        );
    }
    result
}

async fn run(f: &Fixture) -> Result<()> {
    let mut children = (1..=3)
        .map(|id| Ok((id, f.spawn(id, 1)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let leader_a = leader(f, 10).await?;
    let leader_b = leader(f, 11).await?;
    leader(f, 5).await?;
    let table = Uuid::from_u128(50);
    open(f, 10, 0, table).await?;
    open(f, 11, 2, table).await?;
    // Both groups share this physical engine. Disconnect a common follower so
    // both must catch it up while their independent workers keep accepting data.
    let follower = (1..=3)
        .find(|id| *id != leader_a && *id != leader_b)
        .unwrap();
    children.get_mut(&follower).unwrap().kill()?;
    let write = |group, bucket| async move {
        let mut records = Vec::new();
        for sequence in 1..=16 {
            let batch = Batch::new(
                table,
                bucket,
                producer::producer_id(0, 1),
                sequence,
                Bytes::from(vec![(group + sequence) as u8; 256 << 10]),
            );
            let index = accepted(&f.client, group, &batch).await?;
            records.push((group, index, batch));
        }
        Ok::<_, anyhow::Error>(records)
    };
    let (a, b) = tokio::try_join!(write(10, 0), write(11, 2))?;
    let records: Vec<_> = a.into_iter().chain(b).collect();
    // Force checkpoint-backed catch-up for each group's accepted tail.
    for (group, id) in [(10, leader_a), (11, leader_b)] {
        let tail = records
            .iter()
            .filter(|r| r.0 == group)
            .map(|r| r.1)
            .max()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let status = f.status(id).await?;
                if status["groups"][group.to_string()]["checkpoint_index"]
                    .as_u64()
                    .unwrap_or(0)
                    >= tail
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
    }
    children.insert(follower, f.spawn(follower, 2)?);
    for (group, index, batch) in &records {
        for node in 1..=3 {
            fetch(f, node, *group, *index, batch).await?;
        }
    }
    let status = f.status(follower).await?;
    assert!(status["snapshots_installed"].as_u64().unwrap() >= 2);
    for child in children.values_mut() {
        child.kill()?;
    }
    for node in 1..=3 {
        children.insert(node, f.spawn(node, 3)?);
    }
    leader(f, 10).await?;
    leader(f, 11).await?;
    for (group, index, batch) in &records {
        for node in 1..=3 {
            fetch(f, node, *group, *index, batch).await?;
        }
    }
    for group in [10, 11] {
        let (_, index, batch) = records.iter().rev().find(|r| r.0 == group).unwrap();
        assert_eq!(accepted(&f.client, group, batch).await?, *index);
    }
    for node in 1..=3 {
        let status = f.status(node).await?;
        assert_eq!(status["groups"]["5"]["status"]["accepted_records"], 0);
        assert_eq!(status["groups"]["5"]["status"]["producers"], 0);
        for group in ["10", "11"] {
            assert_eq!(status["groups"][group]["status"]["accepted_records"], 16);
        }
        children.get_mut(&node).unwrap().terminate().await?;
    }
    Ok(())
}
