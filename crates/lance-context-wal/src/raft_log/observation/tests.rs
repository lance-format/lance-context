use super::*;
use crate::{placement::Layout, raft_log::PhysicalLog};
use anyhow::Result;
use raft::Storage;
use raft_proto::eraftpb::{Entry, HardState};
use std::collections::BTreeMap;

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
    let writer = std::thread::spawn(move || -> Result<StorageWaitObservation> {
        assert_eq!(snapshot(), StorageWaitObservation::default());
        group.persist(&[], None)?;
        assert_eq!(snapshot(), StorageWaitObservation::default());
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
    let json = serde_json::to_value(observed)?;
    assert_eq!(json["sync"]["attempts"], 1);
    drop(log);
    let reopened = PhysicalLog::open(directory.path(), &layout, 1)?;
    assert_eq!(reopened.group(10)?.initial_state()?.hard_state.commit, 1);
    Ok(())
}
