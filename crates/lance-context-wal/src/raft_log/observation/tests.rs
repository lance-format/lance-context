use super::*;
use crate::{placement::Layout, raft_log::PhysicalLog};
use anyhow::Result;
use raft::Storage;
use raft_proto::eraftpb::{Entry, HardState};
use std::collections::BTreeMap;

#[test]
fn sync_wait_covers_the_interval_after_acquisition_starts() {
    let gate = std::sync::Arc::new(std::sync::Mutex::new(()));
    let held = gate.lock().unwrap();
    let writer_gate = gate.clone();
    let (entered, started) = std::sync::mpsc::channel();
    let writer = std::thread::spawn(move || {
        // Signal inside acquire, after its clock starts. Merely observing the
        // preceding disk append would race a worker not yet at the sync gate.
        drop(acquire(Gate::Sync, || {
            entered.send(()).unwrap();
            writer_gate.lock().unwrap()
        }));
        snapshot()
    });
    started
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let held_since = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(5));
    let held_micros = held_since.elapsed().as_micros() as u64;
    drop(held);
    let observed = writer.join().unwrap();
    assert_eq!(observed.sync.attempts, 1);
    assert!(observed.sync.elapsed_micros >= held_micros);
    assert_eq!(
        observed.sync.elapsed_micros,
        observed.sync.max_elapsed_micros
    );
}

#[test]
fn real_write_counts_each_gate_without_leaking_to_another_thread() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let layout = Layout {
        cluster: uuid::Uuid::from_u128(1),
        nodes: (1..=3).map(|id| (id, format!("wal-{id}:7000"))).collect(),
        groups: BTreeMap::from([(10, vec![1, 2, 3])]),
        buckets: vec![10],
    };
    let log = PhysicalLog::open(directory.path(), &layout, 1)?;
    let group = log.group(10)?;
    let observer_before = snapshot();
    let writes_before = write_snapshot();
    let writer = std::thread::spawn(move || -> Result<StorageWaitObservation> {
        assert_eq!(snapshot(), StorageWaitObservation::default());
        group.persist(&[], None)?;
        assert_eq!(snapshot(), StorageWaitObservation::default());
        assert_eq!(write_snapshot(), StorageWriteObservation::default());
        group.persist(
            &[Entry {
                index: 1,
                term: 1,
                data: b"opaque record".to_vec().into(),
                ..Entry::default()
            }],
            Some(&HardState {
                term: 1,
                commit: 1,
                ..HardState::default()
            }),
        )?;
        assert_eq!(group.initial_state()?.hard_state.commit, 1);
        let writes = write_snapshot();
        assert_eq!(writes.append_batches, 1);
        assert_eq!(writes.sync_barriers, 1);
        assert!(writes.append_bytes >= b"opaque record".len() as u64);
        assert!(writes.sync_marker_bytes > 0);
        // A rejected write never reaches the engine or changes its counters.
        assert!(group.persist(&[], Some(&HardState::default())).is_err());
        assert_eq!(write_snapshot(), writes);
        let observed = snapshot();
        for interval in [
            &observed.maintenance,
            &observed.append,
            &observed.sync,
            &observed.exclusive,
        ] {
            assert_eq!(interval.attempts, 1);
            assert_eq!(interval.elapsed_micros, interval.max_elapsed_micros);
        }
        Ok(observed)
    });
    let observed = writer.join().unwrap()?;
    assert_eq!(snapshot(), observer_before);
    assert_eq!(write_snapshot(), writes_before);
    let json = serde_json::to_value(observed)?;
    assert_eq!(json["sync"]["attempts"], 1);
    drop(log);
    let reopened = PhysicalLog::open(directory.path(), &layout, 1)?;
    assert_eq!(reopened.group(10)?.initial_state()?.hard_state.commit, 1);
    Ok(())
}
