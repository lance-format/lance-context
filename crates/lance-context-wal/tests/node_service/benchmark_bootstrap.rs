use super::*;
use std::fs;

fn launch(
    f: &Fixture,
    node: u64,
    incarnation: u64,
    threads: &str,
    deadline: &str,
) -> Result<fixture::ChildNode> {
    f.spawn_with(
        node,
        incarnation,
        env!("CARGO_BIN_EXE_context-wal-bench-node"),
        &[
            "--leader-plan",
            f.root.path().join("leaders.json").to_str().unwrap(),
            "--ready-file",
            f.root
                .path()
                .join(format!("ready-{node}.json"))
                .to_str()
                .unwrap(),
            "--io-threads",
            threads,
            "--bootstrap-timeout-ms",
            deadline,
        ],
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn benchmark_bootstrap_fixes_placement_then_stops_campaigning() -> Result<()> {
    for (threads, chosen) in [("2", 2), ("4", 2)] {
        let f = Fixture::shared_groups(false)?;
        let result = run(&f, threads, chosen, false).await;
        if result.is_err() {
            eprintln!("failed bootstrap fixture: {}", f.root.keep().display());
        }
        result?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn benchmark_bootstrap_transfers_to_a_late_preferred_node() -> Result<()> {
    for threads in ["2", "4"] {
        let f = Fixture::shared_groups(false)?;
        let result = run(&f, threads, 3, true).await;
        if result.is_err() {
            eprintln!("failed bootstrap fixture: {}", f.root.keep().display());
        }
        result?;
    }
    Ok(())
}

async fn run(f: &Fixture, threads: &str, chosen: u64, late: bool) -> Result<()> {
    fs::write(
        f.root.path().join("leaders.json"),
        serde_json::to_vec(&BTreeMap::from([(5, chosen), (10, chosen), (11, chosen)]))?,
    )?;
    let mut children = BTreeMap::new();
    if late {
        for node in [1, 2, 3].into_iter().filter(|&node| node != chosen) {
            children.insert(node, launch(f, node, 1, threads, "30000")?);
        }
        for group in [5, 10, 11] {
            ensure!(leader(f, group).await? != chosen);
        }
        // Let at least one attempted transfer time out before the target starts.
        // The existing leaders must keep serving Raft instead of exiting on the
        // first different leader observation, as the original launcher did.
        tokio::time::sleep(Duration::from_secs(3)).await;
        for (&node, child) in &mut children {
            child.assert_live()?;
            ensure!(!f.root.path().join(format!("ready-{node}.json")).exists());
        }
        children.insert(chosen, launch(f, chosen, 1, threads, "30000")?);
    } else {
        children.insert(chosen, launch(f, chosen, 1, threads, "30000")?);
        // Start the intended leader first; followers use the same real Service/TLS
        // transport. No manually relayed packets or altered Raft timeouts.
        tokio::time::timeout(Duration::from_secs(10), async {
            while f.status(chosen).await.is_err() {
                children.get_mut(&chosen).unwrap().assert_live()?;
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        for node in [1, 2, 3].into_iter().filter(|&node| node != chosen) {
            children.insert(node, launch(f, node, 1, threads, "30000")?);
        }
    }
    let receipts = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            for child in children.values_mut() {
                child.assert_live()?;
            }
            let receipts = (1..=3)
                .map(|node| {
                    Ok(serde_json::from_slice::<serde_json::Value>(&fs::read(
                        f.root.path().join(format!("ready-{node}.json")),
                    )?)?)
                })
                .collect::<Result<Vec<_>>>();
            if let Ok(receipts) = receipts {
                return Ok::<_, anyhow::Error>(receipts);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await??;
    tokio::time::sleep(Duration::from_secs(3)).await;
    for node in 1..=3 {
        let status = f.status(node).await?;
        for group in [5, 10, 11] {
            let observed = &status["groups"][group.to_string()]["status"];
            assert_eq!(observed["leader"], chosen);
            assert_eq!(
                observed["term"],
                receipts[node as usize - 1]["observed_leader_terms"][group.to_string()][1]
            );
        }
    }
    children.get_mut(&chosen).unwrap().terminate().await?;
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let mut changed = true;
            for node in [1, 2, 3].into_iter().filter(|&node| node != chosen) {
                children.get_mut(&node).unwrap().assert_live()?;
                let status = f.status(node).await?;
                for group in [5, 10, 11] {
                    let leader = status["groups"][group.to_string()]["status"]["leader"]
                        .as_u64()
                        .unwrap_or(0);
                    changed &= leader != 0 && leader != chosen;
                }
            }
            if changed {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await??;
    for child in children.values_mut() {
        child.terminate().await?;
    }
    // A restart uses the production node, never this fresh-store launcher.
    // Retain the original receipt elsewhere so the directory check is exercised.
    fs::rename(
        f.root.path().join(format!("ready-{chosen}.json")),
        f.root.path().join(format!("original-ready-{chosen}.json")),
    )?;
    let mut refused = launch(f, chosen, 2, threads, "30000")?;
    expect_failure(&mut refused, "fresh WAL directory").await?;
    assert!(!f.root.path().join(format!("ready-{chosen}.json")).exists());
    Ok(())
}

async fn expect_failure(child: &mut fixture::ChildNode, message: &str) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Err(error) = child.assert_live() {
                ensure!(error.to_string().contains(message), "{error:#}");
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn benchmark_bootstrap_without_quorum_exits_without_readiness() -> Result<()> {
    let f = Fixture::shared_groups(false)?;
    fs::write(
        f.root.path().join("leaders.json"),
        br#"{"5":2,"10":2,"11":2}"#,
    )?;
    let mut node = launch(&f, 2, 1, "2", "500")?;
    let result = expect_failure(&mut node, "bootstrap deadline exceeded").await;
    assert!(!f.root.path().join("ready-2.json").exists());
    if result.is_err() {
        eprintln!("failed bootstrap fixture: {}", f.root.keep().display());
    }
    result
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn benchmark_bootstrap_cannot_move_a_nonparticipating_leader() -> Result<()> {
    let f = Fixture::shared_groups(false)?;
    let result = async {
        let mut first = f.spawn(1, 1)?;
        let mut second = f.spawn(2, 1)?;
        for group in [5, 10, 11] {
            ensure!([1, 2].contains(&leader(&f, group).await?));
        }
        fs::write(
            f.root.path().join("leaders.json"),
            br#"{"5":3,"10":3,"11":3}"#,
        )?;
        let mut unexpected = launch(&f, 3, 1, "2", "1500")?;
        expect_failure(&mut unexpected, "bootstrap deadline exceeded").await?;
        ensure!(!f.root.path().join("ready-3.json").exists());
        first.terminate().await?;
        second.terminate().await?;
        Ok::<_, anyhow::Error>(())
    }
    .await;
    if result.is_err() {
        eprintln!("failed bootstrap fixture: {}", f.root.keep().display());
    }
    result
}
