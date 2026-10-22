#![cfg(feature = "lab")]

//! Regression expectations for the actual pinned dependency and real files.
//! Run unchanged against the original and any candidate dependency correction.
use anyhow::{ensure, Result};
use raft_engine::{internals::LogQueue, Config, Engine, LogBatch, MessageExt, ReadableSize};
use raft_proto::eraftpb::Entry;
use std::path::Path;
use std::sync::Arc;

struct Index;
impl MessageExt for Index {
    type Entry = Entry;
    fn index(entry: &Entry) -> u64 {
        entry.index
    }
}

fn config(path: &Path, file_size: u64) -> Config {
    Config {
        dir: path.to_string_lossy().into_owned(),
        target_file_size: ReadableSize(file_size),
        purge_threshold: ReadableSize(2 * file_size),
        purge_rewrite_threshold: Some(ReadableSize(file_size)),
        enable_log_recycle: false,
        batch_compression_threshold: ReadableSize(0),
        ..Config::default()
    }
}

fn write(engine: &Engine, through: u64, byte: u8) -> Result<()> {
    for index in 1..=through {
        let mut batch = LogBatch::default();
        batch.add_entries::<Index>(
            1,
            &[Entry {
                index,
                term: 1,
                data: vec![byte; 4096].into(),
                ..Entry::default()
            }],
        )?;
        engine.write(&mut batch, true)?;
    }
    Ok(())
}

fn verify(entry: Entry, byte: u8) -> Result<()> {
    ensure!(entry.data.len() == 4096, "unexpected payload length");
    ensure!(
        entry.data.iter().all(|value| *value == byte),
        "wrong engine payload at index {}: expected byte {}, actual first byte {:?}",
        entry.index,
        byte,
        entry.data.first()
    );
    Ok(())
}

fn fresh_thread_verify(engine: Arc<Engine>, through: u64, byte: u8) -> Result<()> {
    std::thread::spawn(move || {
        for index in 1..=through {
            verify(engine.get_entry::<Index>(1, index)?.unwrap(), byte)?;
        }
        Ok(())
    })
    .join()
    .expect("fresh reader panicked")
}

#[test]
fn scalar_reads_distinguish_two_live_engines() -> Result<()> {
    let root = tempfile::tempdir()?;
    let a = Arc::new(Engine::open(config(&root.path().join("a"), 64 * 1024))?);
    let b = Arc::new(Engine::open(config(&root.path().join("b"), 64 * 1024))?);
    write(&a, 1, 41)?;
    write(&b, 1, 42)?;
    fresh_thread_verify(a.clone(), 1, 41)?;
    fresh_thread_verify(b.clone(), 1, 42)?;
    verify(a.get_entry::<Index>(1, 1)?.unwrap(), 41)?;
    verify(b.get_entry::<Index>(1, 1)?.unwrap(), 42)
}

#[test]
fn batched_reads_distinguish_two_live_engines() -> Result<()> {
    let root = tempfile::tempdir()?;
    let a = Engine::open(config(&root.path().join("a"), 64 * 1024))?;
    let b = Engine::open(config(&root.path().join("b"), 64 * 1024))?;
    write(&a, 1, 51)?;
    write(&b, 1, 52)?;
    let mut entries = Vec::new();
    a.fetch_entries_to::<Index>(1, 1, 2, None, &mut entries)?;
    verify(entries.pop().unwrap(), 51)?;
    b.fetch_entries_to::<Index>(1, 1, 2, None, &mut entries)?;
    verify(entries.pop().unwrap(), 52)
}

#[test]
fn closing_an_engine_does_not_leave_a_cache_alias_for_the_next() -> Result<()> {
    let root = tempfile::tempdir()?;
    let a = Engine::open(config(&root.path().join("a"), 64 * 1024))?;
    write(&a, 1, 61)?;
    verify(a.get_entry::<Index>(1, 1)?.unwrap(), 61)?;
    drop(a);
    let b = Engine::open(config(&root.path().join("b"), 64 * 1024))?;
    write(&b, 1, 62)?;
    verify(b.get_entry::<Index>(1, 1)?.unwrap(), 62)
}

#[test]
fn rewrite_never_persists_another_engines_cached_payload() -> Result<()> {
    let root = tempfile::tempdir()?;
    let a = Arc::new(Engine::open(config(&root.path().join("a"), 4096))?);
    let config_b = config(&root.path().join("b"), 4096);
    let b = Arc::new(Engine::open(config_b.clone())?);
    write(&a, 8, 71)?;
    write(&b, 8, 72)?;
    fresh_thread_verify(b.clone(), 8, 72)?;
    let before = b.file_span(LogQueue::Append);
    ensure!(
        before.1 > before.0 + 2,
        "fixture needs several closed segments"
    );
    verify(a.get_entry::<Index>(1, 1)?.unwrap(), 71)?;
    b.purge_expired_files()?;
    ensure!(
        b.file_span(LogQueue::Append).0 > before.0,
        "test did not exercise physical rewrite/reclamation"
    );
    drop(b);
    let reopened = Arc::new(Engine::open(config_b)?);
    // Reading on a new OS thread distinguishes persisted corruption from a
    // second cache hit in the test's original thread.
    fresh_thread_verify(reopened, 8, 72)
}

#[test]
fn compressed_and_uncompressed_blocks_preserve_all_entry_fields() -> Result<()> {
    let root = tempfile::tempdir()?;
    for compressed in [false, true] {
        let mut options = config(
            &root.path().join(format!("compressed-{compressed}")),
            64 << 10,
        );
        options.batch_compression_threshold = ReadableSize(u64::from(compressed));
        let expected: Vec<_> = (1..=8)
            .map(|index| Entry {
                index,
                term: 4,
                context: vec![index as u8; 129].into(),
                data: vec![index as u8; 1 << 20].into(),
                ..Default::default()
            })
            .collect();
        {
            let engine = Engine::open(options.clone())?;
            let mut batch = LogBatch::default();
            batch.add_entries::<Index>(1, &expected)?;
            engine.write(&mut batch, true)?;
        }
        let engine = Engine::open(options)?;
        let mut actual = Vec::new();
        engine.fetch_entries_to::<Index>(1, 1, 9, None, &mut actual)?;
        ensure!(actual == expected, "batch read changed entry fields");
        for entry in expected.iter().rev() {
            ensure!(engine.get_entry::<Index>(1, entry.index)?.as_ref() == Some(entry));
        }
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn cache_miss_still_checks_the_original_block_checksum() -> Result<()> {
    use std::{fs::OpenOptions, os::unix::fs::FileExt};
    let root = tempfile::tempdir()?;
    let engine = Engine::open(config(root.path(), 64 << 10))?;
    write(&engine, 2, 93)?;
    // Keep the second block cached while corrupting the first block on disk.
    verify(engine.get_entry::<Index>(1, 2)?.unwrap(), 93)?;
    let mut changed = None;
    for item in std::fs::read_dir(root.path())? {
        let path = item?.path();
        if path.extension().is_none_or(|ext| ext != "raftlog") {
            continue;
        }
        let bytes = std::fs::read(&path)?;
        if let Some(offset) = bytes.windows(128).position(|w| w == [93; 128]) {
            let file = OpenOptions::new().read(true).write(true).open(path)?;
            file.write_all_at(&[92], offset as u64)?;
            file.sync_all()?;
            changed = Some((file, offset));
            break;
        }
    }
    let (file, offset) = changed.expect("actual uncompressed entry payload in a log file");
    let error = engine.get_entry::<Index>(1, 1).unwrap_err();
    ensure!(
        error.to_string().contains("Checksum"),
        "unexpected error: {error}"
    );
    // A failed fill must neither install corrupt bytes nor poison the old hit.
    verify(engine.get_entry::<Index>(1, 2)?.unwrap(), 93)?;
    file.write_all_at(&[93], offset as u64)?;
    file.sync_all()?;
    verify(engine.get_entry::<Index>(1, 1)?.unwrap(), 93)
}
