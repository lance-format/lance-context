//! Opt-in storage experiment using the actual node budget and maintenance path.
//! The test filesystem forwards to real files; no fault or pause is enabled.
use super::*;
use crate::admission::DiskBudgetExhausted;
use serde_json::json;
use std::sync::Barrier;
use std::time::{Duration, Instant};

const SEED: u64 = 96;
const WRITES: u64 = 64;
const PAYLOAD: usize = 4 << 20;

fn record(group: u64, index: u64) -> Entry {
    Entry {
        data: vec![(group ^ index) as u8; PAYLOAD].into(),
        ..entry(group, index)
    }
}

fn persist_record(group: &GroupLog, value: &Entry) -> Result<()> {
    group.persist(
        std::slice::from_ref(value),
        Some(&HardState {
            term: 1,
            vote: 1,
            commit: value.index,
            ..Default::default()
        }),
    )
}

#[test]
#[ignore = "writes three fresh one-GiB stores; set WAL_REWRITE_CONCURRENCY_ROOT"]
fn native_rewrite_admission_with_concurrent_durable_appends() -> Result<()> {
    let root = PathBuf::from(std::env::var("WAL_REWRITE_CONCURRENCY_ROOT")?);
    ensure!(!root.exists(), "the experiment requires a fresh directory");
    fs::create_dir(&root)?;
    for (name, rewrite, data_mib) in [
        ("without_rewrite", false, 1280_usize),
        ("bounded_rewrite", true, 1280),
        ("headroom_control", true, 2048),
    ] {
        run_case(&root.join(name), name, rewrite, data_mib << 20)?;
    }
    println!(
        "{}",
        json!({"event":"all_verified","cases":3,"full_payloads_reopened":768,"rf3_lance_qualified":false})
    );
    Ok(())
}

fn run_case(directory: &Path, name: &str, rewrite: bool, data_bytes: usize) -> Result<()> {
    let placement = layout();
    let log = PhysicalLog::open_with_options(
        directory,
        &placement,
        1,
        LogOptions {
            target_file_bytes: 64 << 20,
            purge_threshold_bytes: 256 << 20,
        },
    )?;
    let first = log.group(10)?;
    let second = log.group(20)?;
    for index in 1..=SEED {
        persist_record(&first, &record(10, index))?;
        persist_record(&second, &record(20, index))?;
    }
    first.configure_disk(DiskOptions {
        max_bytes: 3 << 30,
        data_bytes,
        headroom_bytes: 1 << 30,
    })?;
    let budget = first.disk_budget().unwrap().clone();
    let maintenance = first.maintenance();
    // The measured native fixture asks for checkpoints for nine passes before
    // its first forced rewrite. Do not truncate the retained seed to help GC.
    if rewrite {
        for _ in 0..9 {
            maintenance.run()?;
        }
    }
    let before = budget.snapshot();
    let barrier = Arc::new(Barrier::new(2));
    let writer_barrier = barrier.clone();
    let writer_budget = budget.clone();
    let writer = log.group(20)?;
    let started = Instant::now();
    let writer_thread = std::thread::spawn(move || -> Result<serde_json::Value> {
        let mut persisted = 0;
        let mut refused = 0_u64;
        let mut refused_with_rewrite = 0_u64;
        let mut peak_rewrite = 0_u64;
        let mut first_refusal = None;
        let mut last_refusal = None;
        let mut latencies = Vec::new();
        let mut observations = Vec::new();
        writer_barrier.wait();
        while persisted < WRITES {
            ensure!(
                started.elapsed() < Duration::from_secs(60),
                "bounded append window did not drain"
            );
            let observed = writer_budget.snapshot();
            peak_rewrite = peak_rewrite.max(observed.reserved_rewrite_bytes);
            // This covers a native entry plus its small term/HardState update.
            // It is a storage proposal reservation, not a quorum/client ACK.
            let reservation = match writer_budget.reserve(PAYLOAD + (64 << 10), true) {
                Ok(value) => value,
                Err(error) => {
                    let decision = error
                        .downcast_ref::<DiskBudgetExhausted>()
                        .context("unexpected admission failure")?;
                    refused += 1;
                    refused_with_rewrite += u64::from(decision.reserved_rewrite_bytes != 0);
                    peak_rewrite = peak_rewrite.max(decision.reserved_rewrite_bytes);
                    let value =
                        json!({"at_micros":started.elapsed().as_micros(),"decision":decision});
                    if first_refusal.is_none() {
                        first_refusal = Some(value.clone());
                    }
                    last_refusal = Some(value);
                    // Only pre-submission refusals are revisited. An uncertain
                    // persist fails this run; it is never repeated here.
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                }
            };
            let value = record(20, SEED + persisted + 1);
            let clock = Instant::now();
            persist_record(&writer, &value)?;
            latencies.push(clock.elapsed().as_micros() as u64);
            drop(reservation);
            persisted += 1;
            observations.push(json!({"index":value.index,"at_micros":started.elapsed().as_micros(),"reserved_rewrite_bytes":observed.reserved_rewrite_bytes}));
            std::thread::sleep(Duration::from_millis(10));
        }
        latencies.sort_unstable();
        Ok(json!({"persisted":persisted,"refused_attempts":refused,
            "refused_with_rewrite_reservation":refused_with_rewrite,
            "peak_rewrite_reservation_bytes":peak_rewrite,
            "first_refusal":first_refusal,"last_refusal":last_refusal,
            "persist_p50_micros":latencies[latencies.len()/2],
            "persist_p99_micros":latencies[(latencies.len()*99/100).min(latencies.len()-1)],
            "elapsed_micros":started.elapsed().as_micros(),"successful_appends":observations}))
    });
    raft_engine::take_perf_context();
    barrier.wait();
    let maintenance_start = started.elapsed();
    let clock = Instant::now();
    let maintained = if rewrite {
        maintenance.run()
    } else {
        Ok(Vec::new())
    };
    let elapsed = clock.elapsed();
    let maintenance_end = started.elapsed();
    let native = raft_engine::take_perf_context();
    // Always join the only producer before checking either outcome.
    let written = writer_thread
        .join()
        .map_err(|_| anyhow::anyhow!("writer panicked"))?;
    let suggestions = maintained?;
    let written = written?;
    let after = first.disk_measurement()?.run()?;
    ensure!(after.reserved_proposal_bytes == 0 && after.reserved_file_bytes == 0);
    println!(
        "{}",
        json!({"event":"case","case":name,"with_rewrite":rewrite,
        "seed_entries":SEED*2,"payload_bytes":PAYLOAD,"before":before,"after":after,
        "maintenance_start_micros":maintenance_start.as_micros(),"maintenance_end_micros":maintenance_end.as_micros(),
        "maintenance_micros":elapsed.as_micros(),"maintenance_sync_micros":native.log_sync_duration.as_micros(),
        "maintenance_populate_micros":native.log_populating_duration.as_micros(),
        "checkpoint_suggestions":suggestions,"writer":written})
    );
    drop((budget, maintenance, first, second, log));
    let reopened = PhysicalLog::open(directory, &placement, 1)?;
    for (group_id, last) in [(10, SEED), (20, SEED + WRITES)] {
        let group = reopened.group(group_id)?;
        ensure!(group.initial_state()?.hard_state.commit == last);
        for index in 1..=last {
            ensure!(
                group.entries(index, index + 1, None, GetEntriesContext::empty(false))?
                    == vec![record(group_id, index)],
                "recovered payload mismatch: {name}/{group_id}/{index}"
            );
        }
    }
    println!(
        "{}",
        json!({"event":"verified","case":name,"full_payloads_reopened":SEED*2+WRITES,"storage_only":true})
    );
    Ok(())
}
