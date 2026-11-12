#![cfg(all(target_os = "linux", feature = "runner"))]
use anyhow::{ensure, Result};
use std::{
    fs,
    io::{BufRead, BufReader},
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
fn samples_the_original_process_and_finishes_when_it_exits() -> Result<()> {
    for tcp in [false, true] {
        sample_process(tcp)?;
    }
    Ok(())
}

fn sample_process(tcp: bool) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let mut target = OwnedChild(Command::new("sleep").arg("10").spawn()?);
    let stat = fs::read_to_string(format!("/proc/{}/stat", target.0.id()))?;
    let fields: Vec<_> = stat
        .rsplit_once(") ")
        .unwrap()
        .1
        .split_whitespace()
        .collect();
    fs::write(
        directory.path().join("child.json"),
        serde_json::to_vec(&serde_json::json!({
            "pid":target.0.id(),"start_ticks":fields[19].parse::<u64>()?,
            "boot":fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim(),
        }))?,
    )?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_context-wal-resource"));
    command
        .arg("--operation")
        .arg(directory.path())
        .args(["--interval-ms", "100"])
        .stdout(Stdio::piped());
    if tcp {
        command.args(["--tcp-port", "7800", "--threads"]);
    }
    let mut sampler = OwnedChild(command.spawn()?);
    let mut out = BufReader::new(sampler.0.stdout.take().unwrap());
    for _ in 0..2 {
        let mut line = String::new();
        ensure!(out.read_line(&mut line)? > 0, "missing sample");
        let sample: serde_json::Value = serde_json::from_str(&line)?;
        assert_eq!(sample["kind"], "resource_sample");
        assert_eq!(sample["pid"], target.0.id());
        assert!(sample["cgroup_memory_bytes"].as_u64().unwrap() > 0);
        if tcp {
            assert_eq!(sample["tcp"]["port"], 7800);
            assert_eq!(sample["tcp"]["owned_tcp_sockets"], 0);
            assert_eq!(sample["tcp"]["remote_port_states"], serde_json::json!({}));
            let threads = sample["threads"]["observed"].as_array().unwrap();
            assert_eq!(threads.len(), 1);
            assert_eq!(threads[0]["tid"], target.0.id());
            assert_eq!(threads[0]["name"], "sleep");
            assert_eq!(threads[0]["start_ticks"], sample["start_ticks"]);
            assert_eq!(sample["threads"]["vanished"], 0);
        } else {
            assert!(sample["tcp"].is_null());
            assert!(sample["threads"].is_null());
        }
    }
    target.0.kill()?;
    target.0.wait()?;
    let lines: Vec<_> = out.lines().collect::<std::io::Result<_>>()?;
    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap())?;
    assert_eq!(last["kind"], "resource_sampling_complete");
    assert!(last["samples"].as_u64().unwrap() >= 2);
    ensure!(sampler.0.wait()?.success(), "sampler failed");
    Ok(())
}
