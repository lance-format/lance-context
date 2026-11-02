use super::*;
use std::{fs::OpenOptions, os::unix::fs::FileExt};

fn open(directory: &Path) -> Result<Vec<Node>> {
    (1..=3)
        .map(|id| {
            Node::open(
                &directory.join(id.to_string()),
                layout(),
                id,
                Limits::default(),
            )
        })
        .collect()
}

// Change a real stored payload without changing its surrounding metadata or
// checksum. Keep the file open so restoration uses the same inode.
fn corrupt(directory: &Path, extension: &str, byte: u8) -> Result<(std::fs::File, u64)> {
    for item in std::fs::read_dir(directory)? {
        let path = item?.path();
        if path.extension().is_none_or(|ext| ext != extension) {
            continue;
        }
        let bytes = std::fs::read(&path)?;
        if let Some(offset) = bytes.windows(128).position(|w| w == [byte; 128]) {
            let file = OpenOptions::new().read(true).write(true).open(path)?;
            file.write_all_at(&[byte ^ 1], offset as u64)?;
            file.sync_all()?;
            return Ok((file, offset as u64));
        }
    }
    anyhow::bail!("stored payload not found in {extension} files")
}

#[test]
fn native_accepted_reads_check_block_crc_and_record_identity() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut nodes = open(root.path())?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    for sequence in 1..=2 {
        nodes[0].propose(10, sequence, batch(sequence))?;
        pump(&mut nodes)?;
    }
    let entries = nodes[0].accepted(10, 0, 8)?.entries;
    let first = entries[0].0;
    let second = entries[1].0;
    assert!(nodes[0].groups[&10].image.is_none());
    assert_eq!(nodes[0].read_accepted(10, first)?, Some(batch(1)));

    // A valid block still must belong to the captured accepted record.
    nodes[0]
        .groups
        .get_mut(&10)
        .unwrap()
        .state
        .accepted
        .get_mut(&first)
        .unwrap()
        .sequence += 1;
    let error = nodes[0].read_accepted(10, first).unwrap_err();
    assert!(error.to_string().contains("accepted metadata mismatch"));
    nodes[0]
        .groups
        .get_mut(&10)
        .unwrap()
        .state
        .accepted
        .insert(first, entries[0].1.clone());

    // Evict the first block before changing its on-disk bytes.
    assert_eq!(nodes[0].read_accepted(10, second)?, Some(batch(2)));
    let (file, offset) = corrupt(&root.path().join("1"), "raftlog", 1)?;
    let error = nodes[0].read_accepted(10, first).unwrap_err();
    assert!(error.to_string().contains("Checksum"), "{error:#}");
    assert_eq!(nodes[0].read_accepted(10, second)?, Some(batch(2)));
    file.write_all_at(&[1], offset)?;
    file.sync_all()?;
    assert_eq!(nodes[0].read_accepted(10, first)?, Some(batch(1)));
    Ok(())
}

#[test]
fn native_checkpoint_checks_crc_and_captured_metadata_before_install() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut nodes = open(root.path())?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    for sequence in 1..=2 {
        nodes[0].propose(10, sequence, batch(sequence))?;
        pump(&mut nodes)?;
    }
    let entries = nodes[0].accepted(10, 0, 8)?.entries;
    let first = entries[0].0;
    let second = entries[1].0;

    // A non-tail record can have valid checkpoint bookkeeping while its
    // captured identity disagrees with the CRC-validated native entry.
    nodes[0]
        .groups
        .get_mut(&10)
        .unwrap()
        .state
        .accepted
        .get_mut(&first)
        .unwrap()
        .digest[0] ^= 1;
    let error = nodes[0]
        .prepare_checkpoint(10, Uuid::from_u128(811))?
        .run()
        .err()
        .context("checkpoint accepted different captured metadata")?;
    assert!(error.to_string().contains("accepted metadata mismatch"));
    nodes[0]
        .groups
        .get_mut(&10)
        .unwrap()
        .state
        .accepted
        .insert(first, entries[0].1.clone());

    // Force a cache miss so this build reads the damaged native block.
    assert_eq!(nodes[0].read_accepted(10, second)?, Some(batch(2)));
    let (file, offset) = corrupt(&root.path().join("1"), "raftlog", 1)?;
    let error = nodes[0]
        .prepare_checkpoint(10, Uuid::from_u128(812))?
        .run()
        .err()
        .context("checkpoint accepted a damaged native block")?;
    assert!(error.to_string().contains("Checksum"), "{error:#}");
    assert!(nodes[0].groups[&10].image.is_none());
    file.write_all_at(&[1], offset)?;
    file.sync_all()?;
    drop(file);

    let prepared = nodes[0]
        .prepare_checkpoint(10, Uuid::from_u128(813))?
        .run()?;
    nodes[0].install_checkpoint(prepared)?;
    drop(nodes);
    for node in open(root.path())? {
        assert_eq!(node.read_accepted(10, first)?, Some(batch(1)));
        assert_eq!(node.read_accepted(10, second)?, Some(batch(2)));
    }
    Ok(())
}

#[test]
fn image_accepted_reads_keep_digest_checks_and_native_suffix() -> Result<()> {
    let root = tempfile::tempdir()?;
    let mut nodes = open(root.path())?;
    nodes[0].campaign(10)?;
    pump(&mut nodes)?;
    nodes[0].propose(10, 1, batch(1))?;
    pump(&mut nodes)?;
    let prepared = nodes[0]
        .prepare_checkpoint(10, Uuid::from_u128(801))?
        .run()?;
    nodes[0].install_checkpoint(prepared)?;
    nodes[0].propose(10, 2, batch(2))?;
    pump(&mut nodes)?;
    let entries = nodes[0].accepted(10, 0, 8)?.entries;
    let first = entries[0].0;
    let second = entries[1].0;
    let image = nodes[0].groups[&10].image.as_ref().unwrap();
    assert!(image.read_for_validation(first)?.is_some());
    assert!(image.read_for_validation(second)?.is_none());
    assert_eq!(nodes[0].read_accepted(10, first)?, Some(batch(1)));
    assert_eq!(nodes[0].read_accepted(10, second)?, Some(batch(2)));

    let (file, offset) = corrupt(&root.path().join("1"), "pack", 1)?;
    let error = nodes[0].read_accepted(10, first).unwrap_err();
    assert!(
        error.to_string().contains("payload digest mismatch"),
        "{error:#}"
    );
    assert_eq!(nodes[0].read_accepted(10, second)?, Some(batch(2)));
    // Repacking also continues to validate its source bytes.
    let error = nodes[0]
        .prepare_checkpoint(10, Uuid::from_u128(802))?
        .run()
        .err()
        .context("checkpoint accepted a damaged prior image")?;
    assert!(
        error.to_string().contains("payload digest mismatch"),
        "{error:#}"
    );
    file.write_all_at(&[1], offset)?;
    file.sync_all()?;
    drop(file);
    drop(nodes);
    let nodes = open(root.path())?;
    for node in nodes {
        assert_eq!(node.read_accepted(10, first)?, Some(batch(1)));
        assert_eq!(node.read_accepted(10, second)?, Some(batch(2)));
    }
    Ok(())
}
