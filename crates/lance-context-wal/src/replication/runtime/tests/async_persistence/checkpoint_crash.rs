//! Kill the actual runtime before sync, after sync before the owner's receipt,
//! and after checkpoint/suffix completion. A process kill is not a power failure.
use super::*;
use std::{
    os::unix::process::ExitStatusExt,
    path::PathBuf,
    process::{Child, Command as ProcessCommand, Stdio},
};

#[test]
#[ignore = "fixture invoked by the checkpoint SIGKILL test"]
fn checkpoint_child() -> Result<()> {
    let root = PathBuf::from(std::env::var("WAL_METADATA_CRASH_ROOT")?);
    let phase = std::env::var("WAL_METADATA_CRASH_PHASE")?;
    let executor = executor()?;
    let (c, first) = executor.block_on(seeded(&root))?;
    let handle = c.handles[&1].clone();
    let prepared = executor.block_on(async {
        let task = handle.prepare_checkpoint(10, uuid::Uuid::new_v4())?.await?;
        tokio::task::spawn_blocking(move || task.run()).await?
    })?;
    let store = executor.block_on(handle.local(10, |node| {
        Ok(node.groups[&10].raw.store().durable_log().clone())
    })?)?;
    let gate = store.block_sync_for_test();
    let (checkpoint, append) = executor.block_on(async {
        let checkpoint = handle.install_checkpoint(prepared)?;
        wait_for(|| store.appended_checkpoint_for_test().ok().flatten() == Some(first)).await?;
        let append = handle.commit(Command::Append {
            group: 10,
            batch: batch(2),
        })?;
        wait_for(|| handle.observation(10).is_ok_and(|o| o.status.pending == 1)).await?;
        ensure!(checkpoint.0.is_empty() && append.0.is_empty());
        Ok::<_, anyhow::Error>((checkpoint, append))
    })?;
    std::fs::write(root.join("ack"), format!("{first}\n"))?;
    if phase == "before-sync" {
        std::fs::write(root.join("ready"), b"metadata append observed; sync held")?;
    } else if phase == "before-receipt" {
        // Keep the owner in an admitted local operation before releasing sync.
        // The native state lock then proves sync finished, while this operation
        // prevents the owner from consuming its original completion receipt.
        let entered = Arc::new(AtomicBool::new(false));
        let signal = entered.clone();
        let local_store = store.clone();
        let ready = root.join("ready");
        let local = handle.local(10, move |_| {
            signal.store(true, Ordering::Release);
            ensure!(local_store.checkpoint_index()? == first);
            std::fs::write(ready, b"native sync completed; owner receipt pending")?;
            loop {
                std::thread::park();
            }
            #[allow(unreachable_code)]
            Ok(())
        })?;
        executor.block_on(wait_for(|| entered.load(Ordering::Acquire)))?;
        drop(gate);
        // Retain the original local/checkpoint/append handles until SIGKILL.
        std::hint::black_box((&local, &checkpoint, &append));
        loop {
            std::thread::park();
        }
    } else {
        ensure!(phase == "after-ack", "unknown crash phase");
        drop(gate);
        let second = executor.block_on(async {
            checkpoint.await?;
            let second = index(append.await?);
            wait_for(|| {
                c.handles
                    .values()
                    .all(|h| h.observation(10).is_ok_and(|o| o.status.applied >= second))
            })
            .await?;
            Ok::<_, anyhow::Error>(second)
        })?;
        std::fs::write(root.join("ack"), format!("{first}\n{second}\n"))?;
        std::fs::write(root.join("ready"), b"checkpoint and suffix acknowledged")?;
    }
    loop {
        std::thread::park();
    }
}

struct Stop(Child);
impl Drop for Stop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn async_persistence_checkpoint_sigkill_preserves_acks_and_retry_identity() -> Result<()> {
    for phase in ["before-sync", "before-receipt", "after-ack"] {
        let root = if let Some(base) = std::env::var_os("WAL_METADATA_CRASH_ARTIFACTS") {
            std::fs::create_dir_all(&base)?;
            let mut root = tempfile::Builder::new()
                .prefix(&format!("checkpoint-{phase}-"))
                .tempdir_in(base)?;
            root.disable_cleanup(true);
            root
        } else {
            tempfile::tempdir()?
        };
        let stderr = root.path().join("stderr");
        let mut child = Stop(ProcessCommand::new(std::env::current_exe()?)
            .args(["--ignored", "--exact",
                "replication::runtime::tests::async_persistence::checkpoint_crash::checkpoint_child",
                "--nocapture"])
            .env("WAL_METADATA_CRASH_ROOT", root.path())
            .env("WAL_METADATA_CRASH_PHASE", phase)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&stderr)?)
            .spawn()?);
        let until = Instant::now() + Duration::from_secs(20);
        while !root.path().join("ready").exists() {
            ensure!(
                child.0.try_wait()?.is_none() && Instant::now() < until,
                "checkpoint child did not reach {phase}: {}",
                std::fs::read_to_string(&stderr)?
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        child.0.kill()?;
        ensure!(child.0.wait()?.signal() == Some(9), "child was not killed");
        let expected = std::fs::read_to_string(root.path().join("ack"))?
            .lines()
            .enumerate()
            .map(|(i, line)| Ok((line.parse::<u64>()?, i as u64 + 1)))
            .collect::<Result<Vec<_>>>()?;
        reopen(root.path(), &expected)?;
        let node = Node::open(&root.path().join("1"), layout(), 1, Limits::default())?;
        let cut = node.checkpoint_index(10)?;
        ensure!(cut == expected[0].0 || (phase == "before-sync" && cut == 0));
        drop(node);
        let executor = executor()?;
        executor.block_on(async {
            let c = Cluster::open(root.path(), options())?;
            // An unacknowledged suffix may have synced on a follower before
            // the old leader. Raft may correctly choose that follower instead.
            c.handles[&1].campaign(10)?.await?;
            let election = wait_for(|| {
                let Ok(view) = c.handles[&1].observation(10) else {
                    return false;
                };
                view.status.leader != 0
                    && c.handles.values().all(|h| {
                        h.observation(10).is_ok_and(|o| {
                            !o.stopped
                                && o.status.leader == view.status.leader
                                && o.status.term == view.status.term
                        })
                    })
            })
            .await;
            if let Err(error) = election {
                for (id, handle) in c.handles.iter() {
                    eprintln!(
                        "checkpoint_reopen_election: phase={phase} node={id} observation={:?}",
                        handle.observation(10)
                    );
                }
                c.shutdown().await?;
                return Err(error.context(format!(
                    "reopened election at {phase}: {}",
                    root.path().display()
                )));
            }
            let leader = c.handles[&1].observation(10)?.status.leader;
            let second = commit(&c.handles[&leader], 10, batch(2)).await?;
            ensure!(commit(&c.handles[&leader], 10, batch(2)).await? == second);
            if expected.len() == 2 {
                ensure!(second == expected[1].0);
            }
            wait_for(|| {
                c.handles.values().all(|h| {
                    h.observation(10)
                        .is_ok_and(|o| o.status.accepted_records == 2)
                })
            })
            .await?;
            c.shutdown().await?;
            reopen(root.path(), &[(expected[0].0, 1), (second, 2)])
        })?;
        eprintln!("checkpoint_sigkill: phase={phase} signal=9 acknowledged={} full_values=true retry_identity=true", expected.len());
    }
    Ok(())
}
