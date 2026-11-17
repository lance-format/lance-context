//! Read-only Linux experiment sampling of one recorded process incarnation.
//! Same-container cgroup v2 counters are shared, never additive with process RSS.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[path = "resource/tcp.rs"]
mod tcp;
#[path = "resource/threads.rs"]
mod threads;

#[derive(Parser)]
struct Args {
    /// Existing runner operation; this command never launches or stops it.
    #[arg(long)]
    operation: PathBuf,
    #[arg(long, default_value_t = 500)]
    interval_ms: u64,
    #[arg(long)]
    disk_directory: Option<PathBuf>,
    /// Optional TCP socket observations, restricted to this process's own FDs.
    /// Port counts distinguish local endpoints from remote endpoints.
    #[arg(long)]
    tcp_port: Option<u16>,
    /// Include bounded per-thread CPU counters and names for runtime profiling.
    #[arg(long)]
    threads: bool,
}
#[derive(Deserialize)]
struct Identity {
    pid: u32,
    start_ticks: u64,
    boot: String,
}
#[derive(Serialize)]
struct Process {
    state: char,
    flags: u64,
    start_ticks: u64,
    cpu_ticks: u64,
    rss_bytes: u64,
}
impl Process {
    fn is_live_incarnation(&self, identity: &Identity) -> bool {
        // Linux sets PF_EXITING before exit_mm(), while stat may still report R
        // or D. Releasing a large address space need not finish within 100 ms.
        // The flags field is exported by /proc/<pid>/stat (field 9).
        const PF_EXITING: u64 = 0x0000_0004;
        self.start_ticks == identity.start_ticks
            && !matches!(self.state, 'Z' | 'X')
            && self.flags & PF_EXITING == 0
    }
}
fn stat(text: &str, page_bytes: u64) -> Result<Process> {
    let (_, rest) = text.rsplit_once(") ").context("invalid process stat")?;
    let fields: Vec<_> = rest.split_whitespace().collect();
    ensure!(fields.len() >= 22, "truncated process stat");
    Ok(Process {
        state: fields[0].chars().next().context("empty process state")?,
        flags: fields[6].parse()?,
        start_ticks: fields[19].parse()?,
        cpu_ticks: fields[11]
            .parse::<u64>()?
            .checked_add(fields[12].parse()?)
            .context("CPU count overflow")?,
        rss_bytes: fields[21]
            .parse::<u64>()?
            .checked_mul(page_bytes)
            .context("RSS overflow")?,
    })
}
fn counters(text: &str) -> Result<BTreeMap<String, u64>> {
    text.lines()
        .map(|line| {
            let mut fields = line.split_whitespace();
            let key = fields
                .next()
                .context("missing counter name")?
                .trim_end_matches(':')
                .to_owned();
            let value = fields.next().context("missing counter value")?.parse()?;
            ensure!(fields.next().is_none(), "unexpected counter fields");
            Ok((key, value))
        })
        .collect()
}
fn gone(error: &anyhow::Error) -> bool {
    error.downcast_ref::<std::io::Error>().is_some_and(|error| {
        error.kind() == std::io::ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH)
    })
}
fn process(identity: &Identity, page_bytes: u64) -> Result<Option<Process>> {
    let read =
        fs::read_to_string(format!("/proc/{}/stat", identity.pid)).map_err(anyhow::Error::from);
    let body = match read {
        Ok(body) => body,
        Err(error) if gone(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    let process = stat(&body, page_bytes)?;
    if !process.is_live_incarnation(identity) {
        return Ok(None);
    }
    Ok(Some(process))
}
fn finish_sample<T>(sample: Result<T>, identity: &Identity, page_bytes: u64) -> Result<Option<T>> {
    // A dying task can lose access to /proc/<pid>/io before stat reports it as
    // a zombie. Wait briefly for positive evidence that this incarnation is exiting;
    // permission denial alone must never be reported as successful completion.
    let permission_denied = sample.as_ref().err().is_some_and(|error| {
        error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::PermissionDenied)
    });
    let deadline = Instant::now() + Duration::from_millis(100);
    loop {
        if process(identity, page_bytes)?.is_none() {
            return Ok(None);
        }
        if !permission_denied || Instant::now() >= deadline {
            return sample.map(Some);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn allocated(directory: &Path) -> Result<u64> {
    let mut total = 0u64;
    let mut inodes = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = match fs::symlink_metadata(entry.path()) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue, // collection race
            Err(error) => return Err(error.into()),
        };
        ensure!(
            metadata.is_file(),
            "disk sampler requires a flat directory of regular files"
        );
        if !inodes.insert((metadata.dev(), metadata.ino())) {
            continue;
        }
        total = total
            .checked_add(
                metadata
                    .blocks()
                    .checked_mul(512)
                    .context("disk block overflow")?,
            )
            .context("disk total overflow")?;
    }
    Ok(total)
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (100..=10_000).contains(&args.interval_ms),
        "invalid sample interval"
    );
    ensure!(args.tcp_port != Some(0), "invalid TCP observation port");
    let identity: Identity = serde_json::from_slice(&fs::read(args.operation.join("child.json"))?)?;
    ensure!(identity.pid > 0, "invalid PID");
    ensure!(
        fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim() == identity.boot,
        "boot identity changed"
    );
    ensure!(
        fs::read_to_string("/proc/self/cgroup")?.trim() == "0::/",
        "sampler requires a cgroup v2 namespace rooted at its container"
    );
    // A cgroup namespace without a remount can still expose an ancestor's
    // filesystem. Confirm the mounted root contains this process directly.
    let own_pid = std::process::id().to_string();
    ensure!(
        fs::read_to_string("/sys/fs/cgroup/cgroup.procs")
            .context("read mounted cgroup membership")?
            .lines()
            .any(|line| line.trim() == own_pid),
        "cgroup filesystem is not rooted at the sampler's cgroup"
    );
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    ensure!(
        page > 0 && ticks > 0,
        "invalid kernel clock/page configuration"
    );
    let started = Instant::now();
    let interval = Duration::from_millis(args.interval_ms);
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut samples = 0u64;
    loop {
        let begin = Instant::now();
        let Some(initial) = process(&identity, page as u64)? else {
            break;
        };
        let sample = (|| -> Result<serde_json::Value> {
            // Do not accidentally attribute the sampler's container to a process
            // in another cgroup. All experiment samplers run in their target Pod.
            ensure!(
                fs::read_to_string(format!("/proc/{}/cgroup", identity.pid))
                    .context("read target cgroup")?
                    == fs::read_to_string("/proc/self/cgroup").context("read sampler cgroup")?,
                "target and sampler cgroups differ"
            );
            let io = counters(
                &fs::read_to_string(format!("/proc/{}/io", identity.pid))
                    .context("read target I/O counters")?,
            )?;
            let cpu = counters(
                &fs::read_to_string("/sys/fs/cgroup/cpu.stat")
                    .context("read cgroup CPU counters")?,
            )?;
            let memory_events = counters(
                &fs::read_to_string("/sys/fs/cgroup/memory.events")
                    .context("read cgroup memory events")?,
            )?;
            let memory = fs::read_to_string("/sys/fs/cgroup/memory.current")?
                .trim()
                .parse::<u64>()?;
            let disk = args.disk_directory.as_deref().map(allocated).transpose()?;
            let tcp = args
                .tcp_port
                .map(|port| tcp::observe(identity.pid, port))
                .transpose()?;
            let threads = args
                .threads
                .then(|| threads::observe(identity.pid, page as u64))
                .transpose()?;
            Ok(
                serde_json::json!({"kind":"resource_sample", "pid":identity.pid,"start_ticks":identity.start_ticks,
                "timestamp_ms":SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
                "seconds":started.elapsed().as_secs_f64(),"process":initial,"ticks_per_second":ticks,
                "io":io,"cgroup_cpu":cpu,"cgroup_memory_bytes":memory,"cgroup_memory_events":memory_events,
                "allocated_directory_bytes":disk,"tcp":tcp,"threads":threads,
                "sample_wall_seconds":begin.elapsed().as_secs_f64()}),
            )
        })();
        // Ignore incomplete samples only when this incarnation ended or is exiting.
        let Some(sample) = finish_sample(sample, &identity, page as u64)? else {
            break;
        };
        serde_json::to_writer(&mut out, &sample)?;
        out.write_all(b"\n")?;
        out.flush()?;
        samples += 1;
        std::thread::sleep(interval.saturating_sub(begin.elapsed()));
    }
    serde_json::to_writer(
        &mut out,
        &serde_json::json!({"kind":"resource_sampling_complete","samples":samples,
        "seconds":started.elapsed().as_secs_f64(),"reason":"recorded_process_ended_or_exiting"}),
    )?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Read,
        process::{Child, Command, Stdio},
    };

    struct OwnedChild(Child);
    impl Drop for OwnedChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[test]
    #[ignore = "subprocess fixture for real procfs permission transitions"]
    fn permission_fixture() -> Result<()> {
        let ready = std::env::var_os("CONTEXT_RESOURCE_PERMISSION_FIXTURE")
            .context("fixture path missing")?;
        ensure!(unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0) } == 0);
        fs::write(ready, b"ready")?;
        std::io::stdin().read_exact(&mut [0u8])?;
        Ok(())
    }

    fn denied_target() -> Result<(OwnedChild, Identity, anyhow::Error)> {
        let directory = tempfile::tempdir()?;
        let ready = directory.path().join("ready");
        let target = OwnedChild(
            Command::new(std::env::current_exe()?)
                .args(["--exact", "tests::permission_fixture", "--ignored"])
                .env("CONTEXT_RESOURCE_PERMISSION_FIXTURE", &ready)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .spawn()?,
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ready.exists() {
            ensure!(
                Instant::now() < deadline,
                "permission fixture did not start"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        let pid = target.0.id();
        let p = stat(&fs::read_to_string(format!("/proc/{pid}/stat"))?, 4096)?;
        let error = fs::read_to_string(format!("/proc/{pid}/io")).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        Ok((
            target,
            Identity {
                pid,
                start_ticks: p.start_ticks,
                boot: String::new(),
            },
            anyhow::Error::from(error).context("read target I/O counters"),
        ))
    }

    #[test]
    fn permission_denial_during_exit_requires_observed_incarnation_end() -> Result<()> {
        let (mut target, identity, error) = denied_target()?;
        let mut release = target.0.stdin.take().unwrap();
        let exit = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            release.write_all(b"x").unwrap();
        });
        assert!(finish_sample::<()>(Err(error), &identity, 4096)?.is_none());
        exit.join().unwrap();
        ensure!(target.0.wait()?.success(), "fixture failed");
        Ok(())
    }

    #[test]
    fn permission_denial_for_a_live_process_is_an_error() -> Result<()> {
        let (_target, identity, error) = denied_target()?;
        let error = finish_sample::<()>(Err(error), &identity, 4096).unwrap_err();
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert!(process(&identity, 4096)?.is_some());
        Ok(())
    }
    #[test]
    fn kernel_exit_flag_ends_sampling_before_zombie_state() -> Result<()> {
        let identity = Identity {
            pid: 42,
            start_ticks: 99,
            boot: String::new(),
        };
        for state in ['R', 'D', 'S'] {
            // Include another flag to distinguish a bit test from equality.
            for flags in [0x0040_0000, 0x0040_0004] {
                let text = format!(
                    "42 (worker ) x) {state} 1 2 3 4 5 {flags} 0 0 0 0 11 12 0 0 0 0 0 0 99 1024 7"
                );
                let p = stat(&text, 4096)?;
                assert_eq!(p.flags, flags);
                assert_eq!(
                    (p.start_ticks, p.cpu_ticks, p.rss_bytes),
                    (99, 23, 7 * 4096)
                );
                assert_eq!(p.is_live_incarnation(&identity), flags == 0x0040_0000);
            }
        }
        Ok(())
    }
    #[test]
    fn process_stat_handles_spaces_and_parentheses_and_rejects_truncation() -> Result<()> {
        let text = format!(
            "42 (worker ) x) S {} 11 12 0 0 0 0 0 0 99 1024 7",
            ["0"; 10].join(" ")
        );
        let p = stat(&text, 4096)?;
        assert_eq!(
            (p.start_ticks, p.cpu_ticks, p.rss_bytes),
            (99, 23, 7 * 4096)
        );
        assert!(stat("42 (truncated) S 1", 4096).is_err());
        assert_eq!(
            counters("read_bytes: 4\nwrite_bytes: 8\n")?["write_bytes"],
            8
        );
        assert!(counters("bad counter fields").is_err());
        Ok(())
    }
    #[test]
    fn directory_accounting_does_not_follow_links_or_hide_nested_data() -> Result<()> {
        let root = tempfile::tempdir()?;
        let file = root.path().join("data");
        fs::write(&file, vec![1; 8192])?;
        assert_eq!(allocated(root.path())?, fs::metadata(&file)?.blocks() * 512);
        fs::hard_link(&file, root.path().join("hard-link"))?;
        assert_eq!(allocated(root.path())?, fs::metadata(&file)?.blocks() * 512);
        std::os::unix::fs::symlink(&file, root.path().join("link"))?;
        assert!(allocated(root.path()).is_err());
        fs::remove_file(root.path().join("link"))?;
        fs::create_dir(root.path().join("nested"))?;
        assert!(allocated(root.path()).is_err());
        Ok(())
    }
    #[test]
    fn a_reused_pid_is_not_the_recorded_process() -> Result<()> {
        let pid = std::process::id();
        let own = stat(&fs::read_to_string(format!("/proc/{pid}/stat"))?, 4096)?;
        let identity = Identity {
            pid,
            start_ticks: own.start_ticks + 1,
            boot: String::new(),
        };
        assert!(process(&identity, 4096)?.is_none());
        Ok(())
    }
}
