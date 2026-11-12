//! Non-atomic procfs snapshots. Compare counters by (tid, start_ticks), never
//! sum them as total process CPU: threads may exit between observations.
use super::{gone, stat};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::fs;

const MAX_THREADS: usize = 1024;

#[derive(Serialize)]
pub(super) struct Snapshot {
    observed: Vec<Thread>,
    vanished: usize,
}

#[derive(Serialize)]
struct Thread {
    tid: u32,
    name: String,
    state: char,
    start_ticks: u64,
    cpu_ticks: u64,
}

fn parse(tid: u32, text: &str, page_bytes: u64) -> Result<Thread> {
    let (observed_tid, rest) = text.split_once(" (").context("missing thread name")?;
    ensure!(observed_tid.parse::<u32>()? == tid, "thread ID changed");
    let (name, _) = rest.rsplit_once(") ").context("invalid thread name")?;
    let value = stat(text, page_bytes)?;
    Ok(Thread {
        tid,
        name: name.to_owned(),
        state: value.state,
        start_ticks: value.start_ticks,
        cpu_ticks: value.cpu_ticks,
    })
}

pub(super) fn observe(pid: u32, page_bytes: u64) -> Result<Snapshot> {
    let mut snapshot = Snapshot {
        observed: Vec::new(),
        vanished: 0,
    };
    for (count, entry) in fs::read_dir(format!("/proc/{pid}/task"))?.enumerate() {
        ensure!(count < MAX_THREADS, "thread sampling limit exceeded");
        let entry = entry?;
        let tid = entry
            .file_name()
            .to_str()
            .context("invalid thread ID")?
            .parse()?;
        match fs::read_to_string(entry.path().join("stat")) {
            Ok(text) => snapshot.observed.push(parse(tid, &text, page_bytes)?),
            Err(error) => {
                let error = anyhow::Error::from(error);
                if gone(&error) {
                    snapshot.vanished += 1;
                } else {
                    return Err(error.context("read thread stat"));
                }
            }
        }
    }
    snapshot.observed.sort_unstable_by_key(|thread| thread.tid);
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_with_parentheses_keep_correct_identity_and_counters() -> Result<()> {
        let text = "42 (worker ) (cpu) R 1 2 3 4 5 0 7 8 9 10 11 12 13 14 15 16 17 18 1234 20 21";
        let thread = parse(42, text, 4096)?;
        assert_eq!(thread.name, "worker ) (cpu");
        assert_eq!(thread.start_ticks, 1234);
        assert_eq!(thread.cpu_ticks, 23);
        assert!(parse(43, text, 4096).is_err());
        assert!(parse(42, "42 (worker) R 0", 4096).is_err());
        Ok(())
    }

    #[test]
    fn observes_named_thread_incarnation_and_its_exit() -> Result<()> {
        let (ready, receive) = std::sync::mpsc::channel();
        let (stop, stopped) = std::sync::mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("lance-cpu-test".into())
            .spawn(move || {
                ready
                    .send(unsafe { libc::syscall(libc::SYS_gettid) } as u32)
                    .unwrap();
                let _ = stopped.recv();
            })?;
        let tid = receive.recv()?;
        let pid = std::process::id();
        let first = observe(pid, 4096)?;
        let thread = first.observed.iter().find(|t| t.tid == tid).unwrap();
        assert_eq!(thread.name, "lance-cpu-test");
        let second = observe(pid, 4096)?;
        let next = second.observed.iter().find(|t| t.tid == tid).unwrap();
        assert_eq!(next.start_ticks, thread.start_ticks);
        assert!(next.cpu_ticks >= thread.cpu_ticks);
        stop.send(())?;
        worker.join().unwrap();
        // pthread_join can return before the kernel removes the task's procfs entry.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while observe(pid, 4096)?
            .observed
            .iter()
            .any(|t| t.tid == tid && t.start_ticks == thread.start_ticks)
        {
            ensure!(
                std::time::Instant::now() < deadline,
                "joined thread still present"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        Ok(())
    }
}
